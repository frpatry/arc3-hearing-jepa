"""Version rapide (runs + union-find) de explorer.components : même sortie au départage près du pixel cliqué
(le plus proche du centre de l'objet). ~10x plus rapide ; branchée par rollouts.py sur le module explorer."""
import numpy as np


def components(g, bg_frac=0.25, max_n=48):
    g = np.asarray(g)
    H, W = g.shape
    run_lab = np.empty((H, W), np.int64)
    parent = []

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    prev = []
    for y in range(H):
        row = g[y]
        cut = (np.flatnonzero(row[1:] != row[:-1]) + 1).tolist()
        starts, ends = [0] + cut, cut + [W]
        cur, j = [], 0
        for s, e in zip(starts, ends):
            rid = len(parent)
            parent.append(rid)
            col = row[s]
            run_lab[y, s:e] = rid
            while j < len(prev) and prev[j][1] <= s:
                j += 1
            k = j
            while k < len(prev) and prev[k][0] < e:
                if prev[k][2] == col:
                    a, b = find(rid), find(prev[k][3])
                    if a != b:
                        parent[max(a, b)] = min(a, b)
                k += 1
            cur.append((s, e, col, rid))
        prev = cur
    roots = np.array([find(i) for i in range(len(parent))])
    flat = roots[run_lab].ravel()
    uniq, first, lab = np.unique(flat, return_index=True, return_inverse=True)
    n = len(uniq)
    ys, xs = np.divmod(np.arange(H * W), W)
    size = np.bincount(lab, minlength=n)
    cy = np.bincount(lab, ys, n) / size
    cx = np.bincount(lab, xs, n) / size
    d = (ys - cy[lab]) ** 2 + (xs - cx[lab]) ** 2
    order = np.lexsort((np.arange(H * W), d, lab))
    best = order[np.r_[0, np.flatnonzero(np.diff(lab[order])) + 1]]       # pixel le plus proche du centre, par objet
    gf = g.ravel()
    out = [(int(gf[best[k]]), int(size[k]), (int(ys[best[k]]), int(xs[best[k]])), int(first[k]))
           for k in range(n) if size[k] <= bg_frac * H * W]
    out.sort(key=lambda o: (o[1], o[3]))
    uniq_out, keys = [], set()
    for col, sz, (y, x), _ in out:
        kk = (col, y // 2, x // 2)
        if kk not in keys:
            keys.add(kk)
            uniq_out.append((col, sz, (y, x)))
    return uniq_out[:max_n]
