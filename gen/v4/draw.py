"""Dessin « scène officielle » pour les familles v4 (gen/families_v4.py).

Ce qui distingue l'aspect des jeux ARC-AGI-3 officiels d'une grille de tuiles (mesuré dans analysis/MECHANICS.md) :
  - SCÈNES CLAIRSEMÉES : quelques objets (3-12) de taille moyenne (4-12 px) sur un grand fond uniforme ;
    le fond est noir dans ~half des jeux, sinon gris, bleu, bleu clair, blanc, orange… ;
  - PIÈCES / RÉGIONS : grandes zones de couleur (pièce claire posée sur un pourtour plus sombre, deux moitiés de couleurs
    différentes, caverne aux bords irréguliers), barres pleines de couleur vive ;
  - SPRITES DESSINÉS : contours (cadre creux), formes (plus, L, T, barres), sprites multicolores (corps + contour + œil),
    lignes d'1 px, pointillés ;
  - INTERFACE : bandeau en haut (inventaire, cible, compteurs), bandeau en bas (palette), séparateurs, barre de pas d'1 px.
Rien ici n'est recopié d'un jeu officiel : ce sont des générateurs procéduraux de formes et de mises en page.

Couleurs : on dessine directement dans les 16 couleurs officielles décalées de RAW (= 32) pour ne pas entrer en conflit
avec les couleurs sémantiques du noyau (barre, pourtour) ; V4Game.frame() les remet à plat."""
import numpy as np

RAW = 32
WHITE, OFFW, LGREY, GREY, DGREY, BLACK = 0, 1, 2, 3, 4, 5
MAGENTA, PINK, RED, BLUE, LBLUE, YELLOW, ORANGE, MAROON, GREEN, PURPLE = 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
GREYS = (WHITE, OFFW, LGREY, GREY, DGREY, BLACK)
VIVID = (MAGENTA, PINK, RED, BLUE, LBLUE, YELLOW, ORANGE, MAROON, GREEN, PURPLE)

# fond (couleur dominante) : usage mesuré sur les 25 jeux publics (noir ~ 48 %)
BG_P = {BLACK: 0.46, DGREY: 0.08, GREY: 0.08, LGREY: 0.07, OFFW: 0.04, WHITE: 0.06, BLUE: 0.06, LBLUE: 0.08,
        ORANGE: 0.04, YELLOW: 0.01, PURPLE: 0.01, GREEN: 0.01}


def R(c):
    return RAW + int(c)


class Pal:
    """rôles -> couleurs officielles (tirées par jeu). bg = fond, room = sol des pièces, wall = murs / pourtour,
    ui = bandeaux, acc = couleurs vives des objets (distinctes du fond), line = traits fins."""

    def __init__(self, rng, prefer_bg=None):
        ks = list(BG_P)
        p = np.array([BG_P[k] for k in ks])
        self.bg = int(prefer_bg) if prefer_bg is not None else int(rng.choice(ks, p=p / p.sum()))
        dark = self.bg in (BLACK, DGREY, BLUE, PURPLE, MAROON)
        # sol des pièces : un gris voisin (pièce claire sur pourtour sombre, ou l'inverse)
        near = {BLACK: [DGREY, GREY], DGREY: [GREY, BLACK, LGREY], GREY: [LGREY, DGREY, OFFW], LGREY: [GREY, OFFW, WHITE],
                OFFW: [LGREY, WHITE], WHITE: [LGREY, OFFW], BLUE: [LBLUE, BLACK], LBLUE: [WHITE, BLUE],
                ORANGE: [YELLOW, BLACK], YELLOW: [ORANGE, WHITE], PURPLE: [BLACK, MAGENTA], GREEN: [BLACK, LGREY]}
        self.room = int(rng.choice(near.get(self.bg, [GREY])))
        walls = [c for c in (GREY, DGREY, LGREY, WHITE, BLACK, BLUE, LBLUE) if c not in (self.bg, self.room)]
        self.wall = int(rng.choice(walls))
        self.ui = int(rng.choice([c for c in (DGREY, GREY, BLACK, LGREY) if c != self.bg] or [GREY]))
        acc = [c for c in VIVID if c not in (self.bg, self.room)]
        self.acc = [int(c) for c in rng.permutation(acc)]
        self.line = int(rng.choice([c for c in (WHITE, LGREY, GREY, YELLOW, LBLUE) if c not in (self.bg, self.room)]))
        self.fg = WHITE if dark else BLACK                      # contours / détails contrastés
        self.dark = dark
        self.edge = int(rng.choice([c for c in (BLACK, DGREY, GREY, WHITE) if c != self.bg]))

    def a(self, k):
        return self.acc[k % len(self.acc)]


# ----------------------------------------------------------------------------- sprites procéduraux
def _sym_mask(rng, h, w, fill=0.6, vsym=False):
    """masque aléatoire symétrique gauche-droite (et haut-bas si vsym), connexe autour du centre."""
    hw = (w + 1) // 2
    m = rng.random((h, hw)) < fill
    m[h // 2, hw - 1] = True
    full = np.concatenate([m, m[:, :w // 2][:, ::-1]], 1)
    if vsym:
        hh = (h + 1) // 2
        full = np.concatenate([full[:hh], full[:h // 2][::-1]], 0)
    # garder la composante qui contient le centre
    keep = np.zeros_like(full)
    st = [(h // 2, w // 2)]
    full[h // 2, w // 2] = True
    while st:
        y, x = st.pop()
        if 0 <= y < h and 0 <= x < w and full[y, x] and not keep[y, x]:
            keep[y, x] = True
            st += [(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)]
    return keep


def sprite(rng, kind, h, w, c0, c1=None, c2=None):
    """sprite h×w (int, -1 = transparent) ; c0 corps, c1 contour / 2e couleur, c2 détail (œil, point)."""
    s = np.full((h, w), -1, np.int64)
    c1 = c0 if c1 is None else c1
    if kind == "box":                              # aplat plein
        s[:] = c0
    elif kind == "frame":                          # cadre creux (contour d'1 px)
        s[0, :] = s[-1, :] = s[:, 0] = s[:, -1] = c0
    elif kind == "framed":                         # aplat entouré d'un contour d'une autre couleur
        s[:] = c1
        if h > 2 and w > 2:
            s[1:-1, 1:-1] = c0
    elif kind == "plus":
        s[h // 2 - (h // 6):h // 2 + (h // 6) + 1, :] = c0
        s[:, w // 2 - (w // 6):w // 2 + (w // 6) + 1] = c0
    elif kind == "plus_outline":
        t = max(1, h // 3)
        m = np.zeros((h, w), bool)
        m[t:h - t, :] = True
        m[:, t:w - t] = True
        inner = np.zeros_like(m)
        inner[1:-1, 1:-1] = m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:]
        s[m & ~inner] = c0
    elif kind == "ring":
        yy, xx = np.mgrid[:h, :w]
        r = ((yy - (h - 1) / 2) / (h / 2)) ** 2 + ((xx - (w - 1) / 2) / (w / 2)) ** 2
        s[(r <= 1.0) & (r >= 0.45)] = c0
    elif kind == "disc":
        yy, xx = np.mgrid[:h, :w]
        r = ((yy - (h - 1) / 2) / (h / 2)) ** 2 + ((xx - (w - 1) / 2) / (w / 2)) ** 2
        s[r <= 1.05] = c0
    elif kind == "diamond":
        yy, xx = np.mgrid[:h, :w]
        s[np.abs(yy - (h - 1) / 2) / (h / 2) + np.abs(xx - (w - 1) / 2) / (w / 2) <= 1.0] = c0
    elif kind == "blob":                           # forme symétrique aléatoire, contour d'une autre couleur
        m = _sym_mask(rng, h, w, float(rng.uniform(0.5, 0.8)))
        s[m] = c0
        if c1 != c0:
            er = m.copy()
            er[1:-1, 1:-1] = m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:]
            er[0, :] = er[-1, :] = er[:, 0] = er[:, -1] = False
            s[m & ~er] = c1
    elif kind == "glyph":                          # rune : motif binaire 3×3 / 5×5 agrandi
        k = 3 if min(h, w) < 7 else 5
        m = _sym_mask(rng, k, k, 0.55, vsym=bool(rng.random() < 0.3))
        ys = (np.arange(h) * k) // h
        xs = (np.arange(w) * k) // w
        s[m[ys][:, xs]] = c0
    elif kind == "hat":                            # chapeau / bateau : trapèze + tour
        b = max(1, h // 3)
        for y in range(h):
            if y >= h - b:
                s[y, :] = c0
            elif y >= h // 3:
                s[y, w // 4:w - w // 4] = c0
    elif kind == "L":
        t = max(1, min(h, w) // 3)
        s[:, :t] = c0
        s[-t:, :] = c0
    elif kind == "T":
        t = max(1, min(h, w) // 3)
        s[:t, :] = c0
        s[:, w // 2 - t // 2:w // 2 - t // 2 + t] = c0
    elif kind == "bar":
        s[:] = c0
    elif kind == "arrow":
        for y in range(h):
            half = min(w // 2, y) if y < h // 2 + 1 else max(0, w // 6)
            s[y, w // 2 - half:w // 2 + half + 1] = c0
    elif kind == "checker":
        yy, xx = np.mgrid[:h, :w]
        s[:] = np.where((yy + xx) % 2 == 0, c0, c1)
    elif kind == "stripes":
        yy, xx = np.mgrid[:h, :w]
        s[:] = np.where(yy % 2 == 0, c0, c1)
    else:
        s[:] = c0
    if c2 is not None and h >= 3 and w >= 3:       # œil / point de détail
        s[h // 2 - (1 if h >= 6 else 0), w // 2] = c2
    return s


def recolor(s, cmap):
    """remplace les couleurs-gabarits (clés de cmap, ex. 100/101/102) par des couleurs officielles."""
    o = s.copy()
    for k, v in cmap.items():
        o[s == k] = v
    return o


def rot(s, k):
    return np.rot90(s, k)


def blit(g, s, y, x, colour_map=None):
    """colle le sprite s (−1 transparent) en (y, x), coupé aux bords. Couleurs officielles -> RAW+c."""
    h, w = s.shape
    H, W = g.shape
    y0, x0, y1, x1 = max(0, y), max(0, x), min(H, y + h), min(W, x + w)
    if y0 >= y1 or x0 >= x1:
        return
    sub = s[y0 - y:y1 - y, x0 - x:x1 - x]
    m = sub >= 0
    g[y0:y1, x0:x1][m] = RAW + sub[m]


def rect(g, y, x, h, w, c):
    H, W = g.shape
    g[max(0, y):min(H, y + h), max(0, x):min(W, x + w)] = R(c)


def frame_rect(g, y, x, h, w, c, t=1):
    rect(g, y, x, t, w, c)
    rect(g, y + h - t, x, t, w, c)
    rect(g, y, x, h, t, c)
    rect(g, y, x + w - t, h, t, c)


def hline(g, y, x0, x1, c, dotted=0):
    for x in range(min(x0, x1), max(x0, x1) + 1):
        if dotted == 0 or (x - x0) % dotted == 0:
            if 0 <= y < g.shape[0] and 0 <= x < g.shape[1]:
                g[y, x] = R(c)


def vline(g, x, y0, y1, c, dotted=0):
    for y in range(min(y0, y1), max(y0, y1) + 1):
        if dotted == 0 or (y - y0) % dotted == 0:
            if 0 <= y < g.shape[0] and 0 <= x < g.shape[1]:
                g[y, x] = R(c)


def line(g, y0, x0, y1, x1, c, dotted=0):
    """segment de Bresenham (pointillé tous les `dotted` px)."""
    dy, dx = abs(y1 - y0), abs(x1 - x0)
    sy, sx = (1 if y1 > y0 else -1), (1 if x1 > x0 else -1)
    err = dx - dy
    y, x, k = y0, x0, 0
    while True:
        if (dotted == 0 or k % dotted == 0) and 0 <= y < g.shape[0] and 0 <= x < g.shape[1]:
            g[y, x] = R(c)
        if y == y1 and x == x1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
        k += 1


# ----------------------------------------------------------------------------- régions (sol, pièces, cavernes)
def rooms_mask(rng, H, W, n_rooms, min_side=3, max_side=None):
    """union de rectangles reliés (pièces) sur une grille H×W de cases -> bool (True = sol)."""
    max_side = max_side or max(min_side + 1, min(H, W) * 2 // 3)
    m = np.zeros((H, W), bool)
    cy, cx = H // 2, W // 2
    for k in range(n_rooms):
        h = int(rng.integers(min_side, max_side + 1))
        w = int(rng.integers(min_side, max_side + 1))
        if k == 0:
            y = int(np.clip(cy - h // 2 + rng.integers(-2, 3), 0, H - h))
            x = int(np.clip(cx - w // 2 + rng.integers(-2, 3), 0, W - w))
        else:                                                   # accrochée à une case de sol existante
            fr = np.argwhere(m)
            py, px = fr[int(rng.integers(len(fr)))]
            y = int(np.clip(py - int(rng.integers(0, h)), 0, H - h))
            x = int(np.clip(px - int(rng.integers(0, w)), 0, W - w))
        m[y:y + h, x:x + w] = True
    return m


def cave_mask(rng, H, W, fill=0.55, iters=3):
    """caverne aux bords irréguliers (automate cellulaire) ; garde la plus grande composante."""
    m = rng.random((H, W)) < fill
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = False
    for _ in range(iters):
        p = np.pad(m, 1).astype(int)
        nb = sum(p[1 + dy:H + 1 + dy, 1 + dx:W + 1 + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)) - m
        m = (nb >= 5) | (m & (nb >= 4))
        m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = False
    return largest_component(m)


def largest_component(m):
    H, W = m.shape
    seen = np.zeros_like(m)
    best = np.zeros_like(m)
    for y0, x0 in np.argwhere(m):
        if seen[y0, x0]:
            continue
        comp = np.zeros_like(m)
        st = [(y0, x0)]
        while st:
            y, x = st.pop()
            if 0 <= y < H and 0 <= x < W and m[y, x] and not seen[y, x]:
                seen[y, x] = comp[y, x] = True
                st += [(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)]
        if comp.sum() > best.sum():
            best = comp
    return best


def paint_cells(g, mask, T, oy, ox, c):
    for y, x in np.argwhere(mask):
        rect(g, oy + y * T, ox + x * T, T, T, c)


# ----------------------------------------------------------------------------- interface
def hud_strip(g, rng, pal, edge="top", h=None, items=(), dotted_sep=False):
    """bandeau (inventaire / cible / compteur) : fond `pal.ui`, petites cases colorées `items` = [(couleur, plein?)]."""
    H, W = g.shape
    h = h or int(rng.choice([4, 5, 6]))
    y0 = 0 if edge == "top" else H - h
    rect(g, y0, 0, h, W, pal.ui)
    x = 2
    for c, full in items:
        s = h - 2
        if full:
            rect(g, y0 + 1, x, s, s, c)
        else:
            frame_rect(g, y0 + 1, x, s, s, c)
        x += s + 1
    if dotted_sep:
        hline(g, y0 + (h if edge == "top" else -1), 0, W - 1, pal.line, dotted=2)
    return h
