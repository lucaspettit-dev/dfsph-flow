"""Cython core for dfsph_flow.

Implements, in native code:

* SPH smoothing kernels (2D): Poly6 (density) and spiky gradient (pressure).
* Fast quadtree neighborhood search. This is the 2D analog of the "Fast
  Octree Neighborhood Search for SPH Simulations" algorithm
  (Fernandez-Fernandez et al., SIGGRAPH Asia 2022):
    1. Particles are binned into a uniform grid with cell size 1.5 * h
       (support radius) and z-ordered by Morton code (cell-assisted z-sort,
       so each cell is a contiguous block of the particle array).
    2. A quadtree aligned to the grid recursively splits the domain until
       each leaf holds at most ``leaf_cap`` particles.  A leaf's interior
       cells plus the ring of exterior cells within reach of the support
       radius form a self-contained subproblem.
    3. Brute-force distance checks inside each leaf build CSR neighbor
       lists.  Distance comparisons are cheap; the win is a tiny
       acceleration structure, exactly the paper's insight.
* (DFSPH solver appended below once reference formulas are confirmed.)
"""

import numpy as np

from libc.math cimport M_PI, ceil, sqrt
from libc.stdlib cimport qsort

# ---------------------------------------------------------------------------
# SPH kernels (2D)
# ---------------------------------------------------------------------------

cdef inline double _poly6(double r2, double h) nogil:
    """2D Poly6 kernel: W(r,h) = 4/(pi h^8) (h^2 - r^2)^3."""
    cdef double h2 = h * h
    if r2 >= h2:
        return 0.0
    cdef double d = h2 - r2
    cdef double h8 = h2 * h2 * h2 * h2
    return 4.0 / (M_PI * h8) * d * d * d


cdef inline void _spiky_grad(double dx, double dy, double r, double h,
                             double* gx, double* gy) noexcept nogil:
    """Gradient of the 2D spiky kernel.

    W(r,h) = 10/(pi h^5) (h - r)^3  ->  dW/dr = -30/(pi h^5) (h - r)^2.
    grad W = dW/dr * (dx, dy) / r.
    """
    if r >= h or r < 1e-300:
        gx[0] = 0.0
        gy[0] = 0.0
        return
    cdef double q = h - r
    cdef double h5 = h * h * h * h * h
    cdef double c = -30.0 / (M_PI * h5) * q * q / r
    gx[0] = c * dx
    gy[0] = c * dy


# ---------------------------------------------------------------------------
# Morton (Z-order) helpers
# ---------------------------------------------------------------------------

cdef inline unsigned int _part1by1(unsigned int x) nogil:
    x &= 0x0000FFFFu
    x = (x | (x << 8)) & 0x00FF00FFu
    x = (x | (x << 4)) & 0x0F0F0F0Fu
    x = (x | (x << 2)) & 0x33333333u
    x = (x | (x << 1)) & 0x55555555u
    return x


cdef inline unsigned int _morton2(unsigned int x, unsigned int y) nogil:
    return _part1by1(x) | (_part1by1(y) << 1)


cdef struct _CellKey:
    unsigned int key
    int cell


cdef int _cmp_cellkey(const void* a, const void* b) noexcept nogil:
    cdef unsigned int ka = (<_CellKey*>a).key
    cdef unsigned int kb = (<_CellKey*>b).key
    if ka < kb:
        return -1
    if ka > kb:
        return 1
    return 0


# ---------------------------------------------------------------------------
# Neighborhood search
# ---------------------------------------------------------------------------

cdef class NeighborSearch:
    """Quadtree neighborhood search (2D analog of the 2022 octree paper).

    Usage::

        ns = NeighborSearch(h, leaf_cap=1000)
        offsets, indices, perm = ns.build(x, y)

    ``x`` and ``y`` are reordered in place into Morton (z-) order and
    ``perm`` maps new positions to old ones (``x_new[i] == x_old[perm[i]]``),
    so the caller can reorder any other per-particle arrays identically.
    ``offsets``/``indices`` are CSR neighbor lists: neighbors of particle
    ``i`` are ``indices[offsets[i]:offsets[i+1]]``.
    """

    cdef double h
    cdef double cell_size
    cdef int leaf_cap

    def __cinit__(self, double h, int leaf_cap=1000):
        if h <= 0.0:
            raise ValueError("support radius h must be positive")
        if leaf_cap < 1:
            raise ValueError("leaf_cap must be positive")
        self.h = h
        self.cell_size = 1.5 * h  # paper's calibrated optimum
        self.leaf_cap = leaf_cap

    def build(self, double[::1] x, double[::1] y):
        cdef int n = x.shape[0]
        if y.shape[0] != n:
            raise ValueError("x and y must have the same length")
        if n == 0:
            z = np.zeros(0, dtype=np.int64)
            return z, z, z

        cdef double h = self.h
        cdef double cs = self.cell_size
        cdef double h2 = h * h
        cdef int i, c, ix, iy, p, q, L, r, j, s, t
        cdef double dx, dy, d2

        # ---- 1. bounding box + uniform grid -------------------------------
        cdef double xmin = x[0], xmax = x[0], ymin = y[0], ymax = y[0]
        with nogil:
            for i in range(1, n):
                if x[i] < xmin:
                    xmin = x[i]
                elif x[i] > xmax:
                    xmax = x[i]
                if y[i] < ymin:
                    ymin = y[i]
                elif y[i] > ymax:
                    ymax = y[i]
        xmin -= h
        ymin -= h
        xmax += h
        ymax += h
        cdef int nx = <int>ceil((xmax - xmin) / cs)
        cdef int ny = <int>ceil((ymax - ymin) / cs)
        if nx < 1:
            nx = 1
        if ny < 1:
            ny = 1
        # pad to powers of two so quadtree splits stay grid-aligned
        cdef int nx2 = 1
        while nx2 < nx:
            nx2 <<= 1
        cdef int ny2 = 1
        while ny2 < ny:
            ny2 <<= 1
        cdef int ncells = nx * ny

        # ---- 2. bin particles into cells, counting sort -------------------
        cell_of = np.empty(n, dtype=np.int32)
        cell_count = np.zeros(ncells, dtype=np.int32)
        cdef int[::1] cell_of_v = cell_of
        cdef int[::1] cell_count_v = cell_count
        with nogil:
            for i in range(n):
                ix = <int>((x[i] - xmin) / cs)
                if ix < 0:
                    ix = 0
                elif ix >= nx:
                    ix = nx - 1
                iy = <int>((y[i] - ymin) / cs)
                if iy < 0:
                    iy = 0
                elif iy >= ny:
                    iy = ny - 1
                c = iy * nx + ix
                cell_of_v[i] = c
                cell_count_v[c] += 1

        cell_start = np.empty(ncells + 1, dtype=np.int64)
        cdef long[::1] cell_start_v = cell_start
        cdef long acc = 0
        for c in range(ncells):
            cell_start_v[c] = acc
            acc += cell_count_v[c]
        cell_start_v[ncells] = acc

        order = np.empty(n, dtype=np.int64)      # position -> original index
        cursor = cell_start[:-1].copy()
        cdef long[::1] order_v = order
        cdef long[::1] cursor_v = cursor
        with nogil:
            for i in range(n):
                c = cell_of_v[i]
                order_v[cursor_v[c]] = i
                cursor_v[c] += 1

        # reorder x, y by cell (temporary buffers)
        tx = np.empty(n, dtype=np.float64)
        ty = np.empty(n, dtype=np.float64)
        cdef double[::1] tx_v = tx, ty_v = ty
        with nogil:
            for i in range(n):
                tx_v[i] = x[order_v[i]]
                ty_v[i] = y[order_v[i]]
        x[:] = tx_v
        y[:] = ty_v
        # NOTE: order_v maps current positions -> original indices.

        # ---- 3. morton-sort the cells (cell-assisted z-sort) --------------
        key_arr = np.empty(ncells, dtype=np.dtype([("key", "<u4"), ("cell", "<i4")]))
        cdef _CellKey[::1] keys_v = key_arr
        with nogil:
            for c in range(ncells):
                ix = c % nx
                iy = c // nx
                keys_v[c].key = _morton2(<unsigned int>ix, <unsigned int>iy)
                keys_v[c].cell = c
        qsort(&keys_v[0], <size_t>ncells, sizeof(_CellKey), _cmp_cellkey)

        # per morton position: cell coords, particle range, inverse map
        m_ix = np.empty(ncells, dtype=np.int32)
        m_iy = np.empty(ncells, dtype=np.int32)
        m_pstart = np.empty(ncells, dtype=np.int64)
        m_pend = np.empty(ncells, dtype=np.int64)
        inv_morton = np.full(nx2 * ny2, -1, dtype=np.int64)
        cdef int[::1] m_ix_v = m_ix, m_iy_v = m_iy
        cdef long[::1] m_pstart_v = m_pstart, m_pend_v = m_pend
        cdef long[::1] inv_morton_v = inv_morton
        cdef long cum = 0
        cdef unsigned int mc
        for p in range(ncells):
            c = keys_v[p].cell
            ix = c % nx
            iy = c // nx
            m_ix_v[p] = ix
            m_iy_v[p] = iy
            m_pstart_v[p] = cum
            cum += cell_count_v[c]
            m_pend_v[p] = cum
            mc = _morton2(<unsigned int>ix, <unsigned int>iy)
            inv_morton_v[mc] = p
        # prefix particle counts over morton order (for node ranges)
        cumN = np.empty(ncells + 1, dtype=np.int64)
        cdef long[::1] cumN_v = cumN
        cumN_v[0] = 0
        for p in range(ncells):
            cumN_v[p + 1] = m_pend_v[p]

        # final z-order: emit particles cell by cell in morton order.
        # NOTE: the stage-1 array is grouped by cell id, so source ranges
        # come from cell_start (cell-id order), not m_pstart (morton order).
        morton_cell = np.empty(ncells, dtype=np.int32)
        cdef int[::1] morton_cell_v = morton_cell
        for p in range(ncells):
            morton_cell_v[p] = keys_v[p].cell
        perm = np.empty(n, dtype=np.int64)
        cdef long[::1] perm_v = perm
        cdef long dst = 0
        cdef long src, s0
        cdef int cntc
        with nogil:
            for p in range(ncells):
                c = morton_cell_v[p]
                s0 = cell_start_v[c]
                cntc = cell_count_v[c]
                for src in range(s0, s0 + cntc):
                    # src indexes the cell-sorted array; order_v maps to original
                    perm_v[dst] = order_v[src]
                    tx_v[dst] = x[src]
                    ty_v[dst] = y[src]
                    dst += 1
        x[:] = tx_v
        y[:] = ty_v

        # ---- 4. quadtree over the morton-ordered cells --------------------
        cdef int max_nodes = 4 * ncells + 1
        nd_child = np.full((max_nodes, 4), -1, dtype=np.int32)
        nd_cb = np.zeros(max_nodes, dtype=np.int64)
        nd_ce = np.zeros(max_nodes, dtype=np.int64)
        nd_cx0 = np.zeros(max_nodes, dtype=np.int32)
        nd_cy0 = np.zeros(max_nodes, dtype=np.int32)
        nd_cx1 = np.zeros(max_nodes, dtype=np.int32)
        nd_cy1 = np.zeros(max_nodes, dtype=np.int32)
        cdef int[:, ::1] nd_child_v = nd_child
        cdef long[::1] nd_cb_v = nd_cb, nd_ce_v = nd_ce
        cdef int[::1] nd_cx0_v = nd_cx0, nd_cy0_v = nd_cy0
        cdef int[::1] nd_cx1_v = nd_cx1, nd_cy1_v = nd_cy1

        cdef int n_nodes = 1
        nd_cb_v[0] = 0
        nd_ce_v[0] = ncells
        nd_cx0_v[0] = 0
        nd_cy0_v[0] = 0
        nd_cx1_v[0] = nx2
        nd_cy1_v[0] = ny2

        stack = [0]
        leaf_ids = []
        cdef int nid, nchild, k, kk
        cdef int cx0, cy0, cx1, cy1, midx, midy
        cdef long cb, ce, cnt
        # child bboxes: (x0, y0, x1, y1)
        while stack:
            nid = stack.pop()
            cb = nd_cb_v[nid]
            ce = nd_ce_v[nid]
            cx0 = nd_cx0_v[nid]
            cy0 = nd_cy0_v[nid]
            cx1 = nd_cx1_v[nid]
            cy1 = nd_cy1_v[nid]
            cnt = cumN_v[ce] - cumN_v[cb]
            if cnt <= self.leaf_cap or (cx1 - cx0 == 1 and cy1 - cy0 == 1):
                leaf_ids.append(nid)
                continue
            # split the (power-of-two aligned) box
            child_boxes = []
            if cx1 - cx0 > 1 and cy1 - cy0 > 1:
                midx = (cx0 + cx1) // 2
                midy = (cy0 + cy1) // 2
                child_boxes = [(cx0, cy0, midx, midy), (midx, cy0, cx1, midy),
                               (cx0, midy, midx, cy1), (midx, midy, cx1, cy1)]
            elif cx1 - cx0 > 1:
                midx = (cx0 + cx1) // 2
                child_boxes = [(cx0, cy0, midx, cy1), (midx, cy0, cx1, cy1)]
            else:
                midy = (cy0 + cy1) // 2
                child_boxes = [(cx0, cy0, cx1, midy), (cx0, midy, cx1, cy1)]
            nchild = len(child_boxes)
            # one scan over the morton range: children are contiguous runs
            sub_cb = [-1] * nchild
            sub_ce = [-1] * nchild
            cur = -1
            for p in range(cb, ce):
                # find which child box contains this cell
                q = -1
                for k in range(nchild):
                    bx0, by0, bx1, by1 = child_boxes[k]
                    if bx0 <= m_ix_v[p] < bx1 and by0 <= m_iy_v[p] < by1:
                        q = k
                        break
                if q != cur:
                    if cur >= 0:
                        sub_ce[cur] = p
                    if sub_cb[q] < 0:
                        sub_cb[q] = p
                    cur = q
            if cur >= 0:
                sub_ce[cur] = ce
            for k in range(nchild):
                if sub_cb[k] < 0:
                    continue  # empty child (no cells); skip
                bx0, by0, bx1, by1 = child_boxes[k]
                cid = n_nodes
                n_nodes += 1
                nd_child_v[nid, k] = cid
                nd_cb_v[cid] = sub_cb[k]
                nd_ce_v[cid] = sub_ce[k]
                nd_cx0_v[cid] = bx0
                nd_cy0_v[cid] = by0
                nd_cx1_v[cid] = bx1
                nd_cy1_v[cid] = by1
                stack.append(cid)

        cdef int n_leaves = len(leaf_ids)
        leaf_ipb = np.empty(n_leaves, dtype=np.int64)
        leaf_ipe = np.empty(n_leaves, dtype=np.int64)
        leaf_cx0 = np.empty(n_leaves, dtype=np.int32)
        leaf_cy0 = np.empty(n_leaves, dtype=np.int32)
        leaf_cx1 = np.empty(n_leaves, dtype=np.int32)
        leaf_cy1 = np.empty(n_leaves, dtype=np.int32)
        cdef long[::1] leaf_ipb_v = leaf_ipb, leaf_ipe_v = leaf_ipe
        cdef int[::1] leaf_cx0_v = leaf_cx0, leaf_cy0_v = leaf_cy0
        cdef int[::1] leaf_cx1_v = leaf_cx1, leaf_cy1_v = leaf_cy1
        for L in range(n_leaves):
            nid = leaf_ids[L]
            leaf_ipb_v[L] = cumN_v[nd_cb_v[nid]]
            leaf_ipe_v[L] = cumN_v[nd_ce_v[nid]]
            leaf_cx0_v[L] = nd_cx0_v[nid]
            leaf_cy0_v[L] = nd_cy0_v[nid]
            leaf_cx1_v[L] = nd_cx1_v[nid]
            leaf_cy1_v[L] = nd_cy1_v[nid]

        # ---- 5. exterior cell rings per leaf ------------------------------
        cdef int ering = <int>ceil(h / cs)  # cells within reach of h (>=1)
        cdef int ex0, ex1, ey0, ey1
        ext_counts = np.zeros(n_leaves, dtype=np.int64)
        cdef long[::1] ext_counts_v = ext_counts
        for L in range(n_leaves):
            ex0 = leaf_cx0_v[L] - ering
            if ex0 < 0:
                ex0 = 0
            ex1 = leaf_cx1_v[L] + ering
            if ex1 > nx:
                ex1 = nx
            ey0 = leaf_cy0_v[L] - ering
            if ey0 < 0:
                ey0 = 0
            ey1 = leaf_cy1_v[L] + ering
            if ey1 > ny:
                ey1 = ny
            k = 0
            for ix in range(ex0, ex1):
                for iy in range(ey0, ey1):
                    mc = _morton2(<unsigned int>ix, <unsigned int>iy)
                    p = inv_morton_v[mc]
                    if m_pend_v[p] > m_pstart_v[p]:
                        k += 1
            ext_counts_v[L] = k

        ext_off = np.empty(n_leaves + 1, dtype=np.int64)
        cdef long[::1] ext_off_v = ext_off
        acc = 0
        for L in range(n_leaves):
            ext_off_v[L] = acc
            acc += ext_counts_v[L]
        ext_off_v[n_leaves] = acc
        ext_ranges = np.empty(2 * acc, dtype=np.int64)
        cdef long[::1] ext_ranges_v = ext_ranges
        for L in range(n_leaves):
            ex0 = leaf_cx0_v[L] - ering
            if ex0 < 0:
                ex0 = 0
            ex1 = leaf_cx1_v[L] + ering
            if ex1 > nx:
                ex1 = nx
            ey0 = leaf_cy0_v[L] - ering
            if ey0 < 0:
                ey0 = 0
            ey1 = leaf_cy1_v[L] + ering
            if ey1 > ny:
                ey1 = ny
            dst = ext_off_v[L]
            for ix in range(ex0, ex1):
                for iy in range(ey0, ey1):
                    mc = _morton2(<unsigned int>ix, <unsigned int>iy)
                    p = inv_morton_v[mc]
                    if m_pend_v[p] > m_pstart_v[p]:
                        ext_ranges_v[2 * dst] = m_pstart_v[p]
                        ext_ranges_v[2 * dst + 1] = m_pend_v[p]
                        dst += 1

        # ---- 6. brute force per leaf -> CSR neighbor lists ----------------
        nbr_counts = np.zeros(n, dtype=np.int64)
        cdef long[::1] nbr_counts_v = nbr_counts
        cdef long ipb, ipe
        with nogil:
            for L in range(n_leaves):
                ipb = leaf_ipb_v[L]
                ipe = leaf_ipe_v[L]
                for i in range(ipb, ipe):
                    c = 0
                    for r in range(ext_off_v[L], ext_off_v[L + 1]):
                        s = ext_ranges_v[2 * r]
                        t = ext_ranges_v[2 * r + 1]
                        for j in range(s, t):
                            if j == i:
                                continue
                            dx = x[i] - x[j]
                            dy = y[i] - y[j]
                            d2 = dx * dx + dy * dy
                            if d2 <= h2:
                                c += 1
                    nbr_counts_v[i] = c

        offsets = np.empty(n + 1, dtype=np.int64)
        cdef long[::1] offsets_v = offsets
        acc = 0
        for i in range(n):
            offsets_v[i] = acc
            acc += nbr_counts_v[i]
        offsets_v[n] = acc

        indices = np.empty(acc, dtype=np.int64)
        cdef long[::1] indices_v = indices
        fill = offsets[:-1].copy()
        cdef long[::1] fill_v = fill
        with nogil:
            for L in range(n_leaves):
                ipb = leaf_ipb_v[L]
                ipe = leaf_ipe_v[L]
                for i in range(ipb, ipe):
                    for r in range(ext_off_v[L], ext_off_v[L + 1]):
                        s = ext_ranges_v[2 * r]
                        t = ext_ranges_v[2 * r + 1]
                        for j in range(s, t):
                            if j == i:
                                continue
                            dx = x[i] - x[j]
                            dy = y[i] - y[j]
                            d2 = dx * dx + dy * dy
                            if d2 <= h2:
                                indices_v[fill_v[i]] = j
                                fill_v[i] += 1

        return offsets, indices, perm


# ======================================================================
# 2D Divergence-Free SPH solver (Bender & Koschier 2015)
#
# Per-step sequence:
#   1. adaptive dt from CFL condition
#   2. neighborhood search (quadtree/Morton, this module) + z-reorder
#   3. densities rho and pressure-solve factors alpha
#   4. non-pressure forces: gravity + XSPH viscosity
#   5. divergence-free pressure solve (velocity update, signed pressure)
#   6. advect positions, save predicted positions
#   7. neighborhood search + densities/alpha at predicted positions
#   8. constant-density pressure solve (position correction, p >= 0)
#   9. velocity fixup  v += (x - x_adv) / dt
#
# Symmetric pressure acceleration:
#   a_i^p = -sum_j m_j (p_i/rho_i^2 + p_j*/rho_j*^2) grad W_ij
# Boundary neighbors mirror the fluid pressure (p_j* = p_i) and use the
# rest density (rho_j* = rho0) -- the standard sampled-boundary treatment.
#
# Factor alpha_i (diagonal of the density-prediction Jacobian):
#   alpha_i = ( |sum_j m_j grad W_ij|^2 + m_i sum_j m_j |grad W_ij|^2 ) / rho0^2
# with uniform mass m:  alpha_i = m^2 (|sum grad W|^2 + sum |grad W|^2) / rho0^2.
# ======================================================================

cdef class DFSPHSolver:
    """Cython DFSPH solver core. Use ``dfsph_flow.DFSPHFlow`` instead."""

    cdef double dx, h, rho0, mass, gx, gy, xsph_eps, cfl, max_dt
    cdef double tol_dens, tol_div
    cdef int max_iter, leaf_cap
    cdef list _pending
    cdef bint _finalized
    cdef int n, n_fluid
    cdef object _x, _y, _vx, _vy, _rho, _p, _alpha, _drhodt
    cdef object _xadv, _yadv, _isbnd
    cdef object _nbr_off, _nbr_idx
    cdef NeighborSearch _ns

    def __cinit__(self, double dx, double h, double rest_density,
                  double gx, double gy, double xsph_epsilon,
                  double cfl, double max_dt,
                  double max_density_error, double max_divergence_error,
                  int max_iterations, int leaf_cap):
        self.dx = dx
        self.h = h
        self.rho0 = rest_density
        self.mass = rest_density * dx * dx
        self.gx = gx
        self.gy = gy
        self.xsph_eps = xsph_epsilon
        self.cfl = cfl
        self.max_dt = max_dt
        self.tol_dens = max_density_error
        self.tol_div = max_divergence_error
        self.max_iter = max_iterations
        self.leaf_cap = leaf_cap
        self._pending = []
        self._finalized = False
        self.n = 0
        self.n_fluid = 0
        self._ns = NeighborSearch(h, leaf_cap)

    # -- scene construction (buffered until first step) -----------------
    def add_fluid_box(self, double xmin, double ymin, double xmax, double ymax,
                      double vx, double vy):
        self._pending.append(("fluid", xmin, ymin, xmax, ymax, vx, vy))

    def add_boundary_box(self, double xmin, double ymin, double xmax, double ymax):
        self._pending.append(("bbox", xmin, ymin, xmax, ymax, 0.0, 0.0))

    def add_boundary_block(self, double xmin, double ymin, double xmax, double ymax):
        self._pending.append(("block", xmin, ymin, xmax, ymax, 0.0, 0.0))

    def _finalize(self):
        xs, ys, vxs, vys, bnds = [], [], [], [], []
        cdef double d = self.dx

        def grid(x0, y0, x1, y1):
            px = np.arange(x0 + 0.5 * d, x1, d)
            py = np.arange(y0 + 0.5 * d, y1, d)
            if len(px) == 0 or len(py) == 0:
                return
            xx, yy = np.meshgrid(px, py)
            xs.append(xx.ravel()); ys.append(yy.ravel())

        for kind, xmin, ymin, xmax, ymax, vx, vy in self._pending:
            n0 = sum(len(a) for a in xs)
            if kind == "fluid":
                grid(xmin, ymin, xmax, ymax)
            elif kind == "block":
                grid(xmin, ymin, xmax, ymax)
            elif kind == "bbox":
                t = max(2, int(np.ceil(self.h / d))) * d
                grid(xmin - t, ymin - t, xmax + t, ymin)      # bottom wall
                grid(xmin - t, ymax, xmax + t, ymax + t)      # top wall
                grid(xmin - t, ymin, xmin, ymax)              # left wall
                grid(xmax, ymin, xmax + t, ymax)              # right wall
            cnt = sum(len(a) for a in xs) - n0
            vxs.append(np.full(cnt, vx)); vys.append(np.full(cnt, vy))
            bnds.append(np.full(cnt, kind != "fluid", dtype=np.uint8))

        if not xs:
            raise ValueError("DFSPHSolver: no particles added")
        self._x = np.ascontiguousarray(np.concatenate(xs), dtype=np.float64)
        self._y = np.ascontiguousarray(np.concatenate(ys), dtype=np.float64)
        self._vx = np.ascontiguousarray(np.concatenate(vxs), dtype=np.float64)
        self._vy = np.ascontiguousarray(np.concatenate(vys), dtype=np.float64)
        self._isbnd = np.ascontiguousarray(np.concatenate(bnds), dtype=np.uint8)
        self.n = len(self._x)
        self.n_fluid = int(self.n - self._isbnd.sum())
        self._rho = np.full(self.n, self.rho0, dtype=np.float64)
        self._p = np.zeros(self.n, dtype=np.float64)
        self._alpha = np.ones(self.n, dtype=np.float64)
        self._drhodt = np.zeros(self.n, dtype=np.float64)
        self._xadv = self._x.copy()
        self._yadv = self._y.copy()
        self._finalized = True
        self._pending = []

    # -- read-only views ------------------------------------------------
    @property
    def x(self): return self._x
    @property
    def y(self): return self._y
    @property
    def vx(self): return self._vx
    @property
    def vy(self): return self._vy
    @property
    def rho(self): return self._rho
    @property
    def is_boundary(self): return self._isbnd
    @property
    def num_particles(self): return self.n
    @property
    def num_fluid(self): return self.n_fluid

    # -- main step -------------------------------------------------------
    def step(self):
        if not self._finalized:
            self._finalize()
        cdef double dt = self._compute_dt()

        self._rebuild_neighbors()
        self._compute_density_alpha()
        self._non_pressure_forces(dt)
        self._solve_divergence_free(dt)
        self._advect_and_save(dt)

        self._rebuild_neighbors()
        self._compute_density_alpha()
        self._solve_constant_density(dt)
        self._velocity_fixup(dt)
        return dt

    cdef void _advect_and_save(self, double dt):
        cdef double[::1] x = self._x, y = self._y, vx = self._vx, vy = self._vy
        cdef double[::1] xadv = self._xadv, yadv = self._yadv
        cdef int i, n = self.n
        with nogil:
            for i in range(n):
                x[i] += dt * vx[i]
                y[i] += dt * vy[i]
                xadv[i] = x[i]
                yadv[i] = y[i]

    cdef void _velocity_fixup(self, double dt):
        cdef double[::1] x = self._x, y = self._y, vx = self._vx, vy = self._vy
        cdef double[::1] xadv = self._xadv, yadv = self._yadv
        cdef unsigned char[::1] isbnd = self._isbnd
        cdef int i, n = self.n
        with nogil:
            for i in range(n):
                if isbnd[i]:
                    continue
                vx[i] += (x[i] - xadv[i]) / dt
                vy[i] += (y[i] - yadv[i]) / dt

    def debug_half_step(self):
        """Run through the divergence-free solve and return internal arrays."""
        if not self._finalized:
            self._finalize()
        cdef double dt = self._compute_dt()
        self._rebuild_neighbors()
        self._compute_density_alpha()
        out = {
            "dt": dt,
            "rho": self._rho.copy(),
            "alpha": self._alpha.copy(),
            "nbr_count": np.diff(np.asarray(self._nbr_off)),
        }
        self._non_pressure_forces(dt)
        out["v_after_np"] = np.column_stack([self._vx.copy(), self._vy.copy()])
        self._solve_divergence_free(dt)
        out["v_after_divfree"] = np.column_stack([self._vx.copy(), self._vy.copy()])
        out["p"] = self._p.copy()
        out["drhodt"] = self._drhodt.copy()
        self._advect_and_save(dt)
        self._rebuild_neighbors()
        self._compute_density_alpha()
        out["rho2"] = self._rho.copy()
        out["alpha2"] = self._alpha.copy()
        self._solve_constant_density(dt)
        out["rho3"] = self._rho.copy()
        out["p2"] = self._p.copy()
        out["x_after_cd"] = np.column_stack([self._x.copy(), self._y.copy()])
        self._velocity_fixup(dt)
        out["v_final"] = np.column_stack([self._vx.copy(), self._vy.copy()])
        return out

    cdef double _compute_dt(self):
        cdef double[::1] vx = self._vx, vy = self._vy
        cdef unsigned char[::1] isbnd = self._isbnd
        cdef double vmax = 0.0, s
        cdef int i, n = self.n
        with nogil:
            for i in range(n):
                if isbnd[i]:
                    continue
                s = vx[i] * vx[i] + vy[i] * vy[i]
                if s > vmax:
                    vmax = s
        vmax = sqrt(vmax)
        cdef double dt = self.cfl * self.h / (vmax + 1e-9)
        return dt if dt < self.max_dt else self.max_dt

    def _rebuild_neighbors(self):
        off, idx, perm = self._ns.build(self._x, self._y)
        self._vx = np.ascontiguousarray(self._vx[perm])
        self._vy = np.ascontiguousarray(self._vy[perm])
        self._isbnd = np.ascontiguousarray(self._isbnd[perm])
        self._rho = np.ascontiguousarray(self._rho[perm])
        # advected positions (if they exist yet) must follow the same order
        if self._xadv is not None:
            self._xadv = np.ascontiguousarray(self._xadv[perm])
            self._yadv = np.ascontiguousarray(self._yadv[perm])
        self._nbr_off = off
        self._nbr_idx = idx

    cdef void _compute_density_alpha(self):
        """SPH densities and the pressure-solve factor alpha (all particles)."""
        cdef double[::1] x = self._x, y = self._y, rho = self._rho, alpha = self._alpha
        cdef long[::1] off = self._nbr_off, idx = self._nbr_idx
        cdef unsigned char[::1] isbnd = self._isbnd
        cdef double h = self.h, m = self.mass, rho0 = self.rho0
        cdef double dx_, dy_, r, w, gx_, gy_, gx1, gy1
        cdef double rs, qs, ri2
        cdef long i, k, j, s, e, n = self.n
        with nogil:
            for i in range(n):
                rs = 0.0
                gx_ = 0.0; gy_ = 0.0; qs = 0.0
                s = off[i]; e = off[i + 1]
                for k in range(s, e):
                    j = idx[k]
                    dx_ = x[i] - x[j]
                    dy_ = y[i] - y[j]
                    rs += _poly6(dx_ * dx_ + dy_ * dy_, h)
                    r = sqrt(dx_ * dx_ + dy_ * dy_)
                    if r > 1e-12 and r < h:
                        _spiky_grad(dx_, dy_, r, h, &gx1, &gy1)
                        gx_ += gx1
                        gy_ += gy1
                        # only fluid neighbors respond to pressure (walls are fixed),
                        # so only they contribute to the |grad W|^2 response term
                        if not isbnd[j]:
                            qs += gx1 * gx1 + gy1 * gy1
                rho[i] = m * rs
                # use rho[i] (not rho0) so alpha is consistent with the
                # 1/rho[i]^2 scaling used in the pressure velocity update
                ri2 = rho[i] * rho[i]
                if ri2 < 1e-12:
                    ri2 = 1e-12
                alpha[i] = m * m * (gx_ * gx_ + gy_ * gy_ + qs) / ri2
                if alpha[i] < 1e-12:
                    alpha[i] = 1e-12

    cdef void _compute_densities_only(self):
        cdef double[::1] x = self._x, y = self._y, rho = self._rho
        cdef long[::1] off = self._nbr_off, idx = self._nbr_idx
        cdef double h = self.h, m = self.mass
        cdef double dx_, dy_, rs
        cdef long i, k, j, s, e, n = self.n
        with nogil:
            for i in range(n):
                rs = 0.0
                s = off[i]; e = off[i + 1]
                for k in range(s, e):
                    j = idx[k]
                    dx_ = x[i] - x[j]
                    dy_ = y[i] - y[j]
                    rs += _poly6(dx_ * dx_ + dy_ * dy_, h)
                rho[i] = m * rs

    cdef void _non_pressure_forces(self, double dt):
        """Gravity + XSPH viscosity on velocities."""
        cdef double[::1] x = self._x, y = self._y, vx = self._vx, vy = self._vy
        cdef double[::1] rho = self._rho
        cdef long[::1] off = self._nbr_off, idx = self._nbr_idx
        cdef unsigned char[::1] isbnd = self._isbnd
        cdef double gx = self.gx, gy = self.gy, eps = self.xsph_eps
        cdef double h = self.h, m = self.mass
        cdef double dx_, dy_, r, w, corr, cx, cy
        cdef long i, k, j, s, e, n = self.n
        with nogil:
            for i in range(n):
                if isbnd[i]:
                    continue
                vx[i] += dt * gx
                vy[i] += dt * gy
                if eps > 0.0:
                    cx = 0.0; cy = 0.0
                    s = off[i]; e = off[i + 1]
                    for k in range(s, e):
                        j = idx[k]
                        if j == i:
                            continue
                        dx_ = x[i] - x[j]
                        dy_ = y[i] - y[j]
                        r = sqrt(dx_ * dx_ + dy_ * dy_)
                        if r < 1e-12 or r >= h:
                            continue
                        w = _poly6(dx_ * dx_ + dy_ * dy_, h)
                        corr = 2.0 * m / (rho[i] + rho[j])
                        cx += corr * (vx[j] - vx[i]) * w
                        cy += corr * (vy[j] - vy[i]) * w
                    vx[i] += eps * cx
                    vy[i] += eps * cy

    cdef void _solve_divergence_free(self, double dt):
        """Enforce d rho/dt = 0 (velocity update, signed pressure allowed)."""
        cdef double[::1] x = self._x, y = self._y, vx = self._vx, vy = self._vy
        cdef double[::1] rho = self._rho, p = self._p, alpha = self._alpha
        cdef double[::1] drhodt = self._drhodt
        cdef long[::1] off = self._nbr_off, idx = self._nbr_idx
        cdef unsigned char[::1] isbnd = self._isbnd
        cdef double h = self.h, m = self.mass, rho0 = self.rho0
        cdef double rho02 = rho0 * rho0
        cdef double dx_, dy_, r, gx_, gy_
        cdef double s, maxd, pi_, pj_, rhoi2, ax, ay
        cdef long i, k, j, s0, e0, n = self.n
        cdef int it
        for it in range(self.max_iter):
            maxd = 0.0
            with nogil:
                for i in range(n):
                    if isbnd[i]:
                        drhodt[i] = 0.0
                        continue
                    s = 0.0
                    s0 = off[i]; e0 = off[i + 1]
                    for k in range(s0, e0):
                        j = idx[k]
                        if j == i:
                            continue
                        dx_ = x[i] - x[j]
                        dy_ = y[i] - y[j]
                        r = sqrt(dx_ * dx_ + dy_ * dy_)
                        if r < 1e-12 or r >= h:
                            continue
                        _spiky_grad(dx_, dy_, r, h, &gx_, &gy_)
                        s += m * ((vx[i] - vx[j]) * gx_ + (vy[i] - vy[j]) * gy_)
                    drhodt[i] = s
                    if s < 0: s = -s
                    if s > maxd: maxd = s
            if maxd * dt / rho0 < self.tol_div:
                break
            with nogil:
                for i in range(n):
                    if isbnd[i]:
                        continue
                    # under-relaxed Jacobi (omega=0.5) for stability;
                    # the pressure matrix is not diagonally dominant.
                    # Clamp p >= 0 (SPlisHSPlasH: only compression is corrected
                    # in the divergence-free solve; expansion handled by
                    # the constant-density solve).
                    p[i] = 0.5 * drhodt[i] / (alpha[i] * dt)
                    if p[i] < 0.0:
                        p[i] = 0.0
                for i in range(n):
                    if isbnd[i]:
                        continue
                    pi_ = p[i]
                    rhoi2 = rho[i] * rho[i]
                    ax = 0.0; ay = 0.0
                    s0 = off[i]; e0 = off[i + 1]
                    for k in range(s0, e0):
                        j = idx[k]
                        if j == i:
                            continue
                        dx_ = x[i] - x[j]
                        dy_ = y[i] - y[j]
                        r = sqrt(dx_ * dx_ + dy_ * dy_)
                        if r < 1e-12 or r >= h:
                            continue
                        _spiky_grad(dx_, dy_, r, h, &gx_, &gy_)
                        if isbnd[j]:
                            # walls have no pressure in the div-free solve;
                            # the fluid's own pressure still pushes against them
                            ax += m * (pi_ / rhoi2) * gx_
                            ay += m * (pi_ / rhoi2) * gy_
                        else:
                            pj_ = p[j]
                            ax += m * (pi_ / rhoi2 + pj_ / (rho[j] * rho[j])) * gx_
                            ay += m * (pi_ / rhoi2 + pj_ / (rho[j] * rho[j])) * gy_
                    vx[i] -= dt * ax
                    vy[i] -= dt * ay

    cdef void _solve_constant_density(self, double dt):
        """Enforce rho = rho0 (position correction, p clamped >= 0)."""
        cdef double[::1] x = self._x, y = self._y, vx = self._vx, vy = self._vy
        cdef double[::1] rho = self._rho, p = self._p, alpha = self._alpha
        cdef long[::1] off = self._nbr_off, idx = self._nbr_idx
        cdef unsigned char[::1] isbnd = self._isbnd
        cdef double h = self.h, m = self.mass, rho0 = self.rho0
        cdef double rho02 = rho0 * rho0
        cdef double dt2 = dt * dt
        cdef double dx_, dy_, r, gx_, gy_
        cdef double err, pi_, pj_, rhoi2, ax, ay
        cdef long i, k, j, s0, e0, n = self.n
        cdef int it
        for it in range(self.max_iter):
            err = 0.0
            with nogil:
                for i in range(n):
                    if isbnd[i]:
                        continue
                    if rho[i] > rho0:
                        q = (rho[i] - rho0) / rho0
                        if q > err:
                            err = q
            if err < self.tol_dens:
                break
            with nogil:
                for i in range(n):
                    if isbnd[i]:
                        continue
                    q = rho[i] - rho0
                    p[i] = q / (alpha[i] * dt2) if q > 0.0 else 0.0
                for i in range(n):
                    if isbnd[i]:
                        continue
                    pi_ = p[i]
                    if pi_ == 0.0:
                        continue
                    rhoi2 = rho[i] * rho[i]
                    ax = 0.0; ay = 0.0
                    s0 = off[i]; e0 = off[i + 1]
                    for k in range(s0, e0):
                        j = idx[k]
                        if j == i:
                            continue
                        dx_ = x[i] - x[j]
                        dy_ = y[i] - y[j]
                        r = sqrt(dx_ * dx_ + dy_ * dy_)
                        if r < 1e-12 or r >= h:
                            continue
                        _spiky_grad(dx_, dy_, r, h, &gx_, &gy_)
                        if isbnd[j]:
                            pj_ = pi_
                            ax += m * (pi_ / rhoi2 + pj_ / rho02) * gx_
                            ay += m * (pi_ / rhoi2 + pj_ / rho02) * gy_
                        else:
                            pj_ = p[j]
                            ax += m * (pi_ / rhoi2 + pj_ / (rho[j] * rho[j])) * gx_
                            ay += m * (pi_ / rhoi2 + pj_ / (rho[j] * rho[j])) * gy_
                    x[i] -= dt2 * ax
                    y[i] -= dt2 * ay
            self._compute_densities_only()
