# -*- coding: utf-8 -*-
# mosaic_lab_pro.py — WERSJA SKONSOLIDOWANA
#  - Siatka 14-portowa (S/H), scena 3D, A*, wejście z klawiatury
#  - 5 zakładek: Mosaic Lab+, Meta-świat (TO), Przekroje 2D, Symulacje (S/H), AST Lab
#
# Uruchomienie:
#   python mosaic_lab_pro.py
#
# Wymagania: Python 3.8+, Tkinter, matplotlib

import ast
import math
import heapq
import tkinter as tk
from tkinter import ttk, filedialog

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ============================================================
# 1) STAŁE I NARZĘDZIA GEOMETRII (14 portów: 6 S, 8 H)
# ============================================================

H_VECS = [(sx, sy, sz) for sx in (+1, -1) for sy in (+1, -1) for sz in (+1, -1)]  # 8 przekątnych
S_VECS = [(+2, 0, 0), (-2, 0, 0), (0, +2, 0), (0, -2, 0), (0, 0, +2), (0, 0, -2)]  # 6 osiowych


def add(a, b): return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a, b): return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def parity_ok(p): return (p[0] & 1) == (p[1] & 1) == (p[2] & 1)


def edge_type(u, v):
    d = sub(v, u)
    if d in H_VECS: return "H"
    if d in S_VECS: return "S"
    return "?"


def parse_point_text(text):
    s = (text or "").strip()
    if not s: raise ValueError("pusta wartość")
    for ch in "[](){}": s = s.replace(ch, " ")
    parts = [p for p in s.replace(",", " ").split() if p]
    if len(parts) != 3: raise ValueError("wymagane trzy liczby")
    x, y, z = map(int, parts)
    return (x, y, z)


def project_to_l2(p):
    """Najbliższy (w L1) punkt kratownicy x≡y≡z (mod 2)."""
    x, y, z = p
    px = x & 1
    if (y & 1) != px: y += 1 if ((y + 1) & 1) == px else -1
    if (z & 1) != px: z += 1 if ((z + 1) & 1) == px else -1
    return (x, y, z)


def sgn(v): return 1 if v > 0 else (-1 if v < 0 else 0)


# ===== META: narzędzia =====

def lerp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t)


def bbox_of(points):
    if not points:
        return (0, 0, 0, 0, 0, 0)
    xs = [p[0] for p in points];
    ys = [p[1] for p in points];
    zs = [p[2] for p in points]
    return (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))


class DSU:
    def __init__(self):
        self.p = {}

    def find(self, x):
        if x not in self.p: self.p[x] = x
        if self.p[x] != x: self.p[x] = self.find(self.p[x])
        return self.p[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[rb] = ra


# ============================================================
# 2) PRAWA I PLANOWANIE (A*)
# ============================================================

class Laws:
    """Koszt kroku i heurystyka A*; parametryzowalne wagami wH, wS.
       Heurystyka gwarantuje admissibility i consistency dla siatki 14-portowej.
    """

    def __init__(self, wH=1.0, wS=1.0, heuristic="mix"):
        self.wH = float(wH)
        self.wS = float(wS)
        # heuristic: "mix" (dokładniejsza relaksacja liniowa + parytet) | "l1" (prosty LB) | "zero"
        self.heuristic = heuristic

    def step_cost(self, u, v):
        t = edge_type(u, v)
        if t == "H": return self.wH
        if t == "S": return self.wS
        return float("inf")

    @staticmethod
    def _l1(u, v):
        return abs(v[0] - u[0]) + abs(v[1] - u[1]) + abs(v[2] - u[2])

    def _lb_l1_relax(self, u, v):
        """Dolne ograniczenie przez ciągłą relaksację L1:
           - krok H może zredukować L1 maks. o 3,
           - krok S może zredukować L1 maks. o 2.
           LB = min( (L1/3)*wH, (L1/2)*wS ).
           (Admissible i consistent; nie uwzględnia parytetu, ale nie przeszacuje.)
        """
        L1 = self._l1(u, v)
        return min((L1 / 3.0) * self.wH, (L1 / 2.0) * self.wS)

    def _lb_mixed_with_parity(self, u, v):
        """Ściślejszy LB: wybiera tańszą 'gęstość redukcji L1' (wH/3 vs wS/2),
           ale koryguje pod parytet (liczba kroków H musi być parzysta).
        """
        L1 = self._l1(u, v)

        # gęstość kosztu: koszt za jednostkę redukcji L1
        dens_H = self.wH / 3.0
        dens_S = self.wS / 2.0

        if dens_H <= dens_S:
            # Preferuj H: minimalnie ceil(L1/3) kroków H. Wymuś parzystość.
            k = math.ceil(L1 / 3.0)
            if k % 2 == 1:
                k += 1  # parzystość (H zmienia parytet wszystkich współrzędnych)
            # Jeśli po tych H zostało coś z L1 (przez parytet), domknij najtańszym środkiem
            reduced = 3 * k
            if reduced >= L1:
                return k * self.wH
            rem = L1 - reduced
            # Domknięcie S (bez naruszenia admissibility)
            s = math.ceil(rem / 2.0)
            return k * self.wH + s * self.wS
        else:
            # Preferuj S: wystarczy ceil(L1/2) kroków S jako LB.
            s = math.ceil(L1 / 2.0)
            return s * self.wS

    def h_estimate(self, u, goal):
        if self.heuristic == "zero":
            return 0.0
        if self.heuristic == "l1":
            return self._lb_l1_relax(u, goal)
        # "mix": dokładniejszy i nadal admissible/consistent
        return self._lb_mixed_with_parity(u, goal)


def neighbors_weighted(u, laws):
    for d in H_VECS + S_VECS:
        v = add(u, d)
        if parity_ok(v):
            yield v, laws.step_cost(u, v)


def validate_path(path):
    """Sprawdza:
       - parytet każdego węzła (x≡y≡z mod 2),
       - czy każde (u,v) jest dopuszczalnym krokiem H lub S.
       Zwraca (ok, msg).
    """
    if not path or len(path) < 2:
        return False, "Ścieżka pusta."
    for p in path:
        if not parity_ok(p):
            return False, f"Naruszony parytet w punkcie {p}."
    for u, v in zip(path, path[1:]):
        t = edge_type(u, v)
        if t not in ("H", "S"):
            return False, f"Niedozwolony krok {u}→{v} (Δ={sub(v, u)})."
    return True, "OK"


def astar(start, goal, laws, max_nodes=25000):
    R = max(8, max(abs(start[0] - goal[0]), abs(start[1] - goal[1]), abs(start[2] - goal[2])) + 8)

    def inside(p):
        return (abs(p[0] - start[0]) <= R and abs(p[1] - start[1]) <= R and abs(p[2] - start[2]) <= R)

    open_heap = []
    heapq.heappush(open_heap, (laws.h_estimate(start, goal), 0.0, start, None))
    came = {}
    best_g = {start: 0.0}
    seen = 0

    while open_heap and seen < max_nodes:
        f, g, u, parent = heapq.heappop(open_heap)
        seen += 1
        if u in came:  # closed
            continue
        came[u] = parent
        if u == goal:
            path = [u]
            while came[u] is not None:
                u = came[u];
                path.append(u)
            path.reverse()
            return path, g, seen
        for v, w in neighbors_weighted(u, laws):
            if not inside(v): continue
            ng = g + w
            if v not in best_g or ng < best_g[v]:
                best_g[v] = ng
                nf = ng + laws.h_estimate(v, goal)
                heapq.heappush(open_heap, (nf, ng, v, u))
    return None, float("inf"), seen


# ============================================================
# 3) SCENA 3D
# ============================================================

class Scene3D:
    def __init__(self, master, title=""):
        self.fig = plt.Figure(figsize=(7.6, 7.6))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, master);
        self.toolbar.update()
        self.title = title
        self.origin = (0, 0, 0)
        self._draw_origin_gizmo = True
        self._draw_compass = False
        self.reset_view()

    def reset_view(self):
        self.ax.view_init(elev=20, azim=-60)
        self.ax.set_xlabel("X");
        self.ax.set_ylabel("Y");
        self.ax.set_zlabel("Z")
        self.ax.set_title(self.title or "")
        self.canvas.draw_idle()

    def clear(self, title=None):
        self.ax.clear()
        self.ax.set_xlabel("X");
        self.ax.set_ylabel("Y");
        self.ax.set_zlabel("Z")
        self.ax.set_title(title or self.title or "")
        if self._draw_compass: self.draw_compass()
        if self._draw_origin_gizmo: self.draw_origin()

    def set_limits(self, xs, ys, zs):
        xmax = max(xs + [6]);
        ymax = max(ys + [6]);
        zmax = max(zs + [6])
        self.ax.set_xlim(min(xs + [-6]) - 1, xmax + 1)
        self.ax.set_ylim(min(ys + [-6]) - 1, ymax + 1)
        self.ax.set_zlim(min(zs + [-6]) - 1, zmax + 1)

    def set_origin(self, p):
        self.origin = p

    def draw_origin(self, size=1.2):
        ox, oy, oz = self.origin
        self.ax.plot([ox - size, ox + size], [oy, oy], [oz, oz], linewidth=1.4)
        self.ax.plot([ox, ox], [oy - size, oy + size], [oz, oz], linewidth=1.4)
        self.ax.plot([ox, ox], [oy, oy], [oz - size, oz + size], linewidth=1.4)
        self.ax.text(ox + size * 1.05, oy, oz, "O", fontsize=10)

    def draw_compass(self):
        ox, oy, oz = self.origin
        self.ax.plot([ox, ox + 6], [oy, oy], [oz, oz]);
        self.ax.text(ox + 6.2, oy, oz, "X")
        self.ax.plot([ox, ox], [oy, oy + 6], [oz, oz]);
        self.ax.text(ox, oy + 6.2, oz, "Y")
        self.ax.plot([ox, ox], [oy, oy], [oz, oz + 6]);
        self.ax.text(ox, oy, oz + 6.2, "Z")
        for sx, sy, sz in S_VECS:
            self.ax.plot([ox, ox + sx], [oy, oy + sy], [oz, oz + sz], linewidth=1.0, alpha=0.9)
        for hx, hy, hz in H_VECS:
            self.ax.plot([ox, ox + hx], [oy, oy + hy], [oz, oz + hz], linewidth=0.9, alpha=0.8)

    def draw_points(self, pts, size=8):
        if not pts: return
        xs = [p[0] for p in pts];
        ys = [p[1] for p in pts];
        zs = [p[2] for p in pts]
        self.ax.scatter(xs, ys, zs, s=size)

    def draw_edge_S(self, u, v, lw=1.6):
        (x1, y1, z1), (x2, y2, z2) = u, v
        self.ax.plot([x1, x1], [y1, y1], [z1, z2], linewidth=lw)
        if x1 != x2: self.ax.plot([x1, x2], [y1, y1], [z2, z2], linewidth=lw)
        if y1 != y2: self.ax.plot([x2, x2], [y1, y2], [z2, z2], linewidth=lw)

    def draw_edge_H(self, u, v, lw=1.4):
        (x1, y1, z1), (x2, y2, z2) = u, v
        self.ax.plot([x1, x2], [y1, y2], [z1, z2], linewidth=lw)

    def _axis_index(self, axis: str) -> int:
        """Mapuje nazwę osi na indeks współrzędnej."""
        a = (axis or "Z").upper()
        if a == "X": return 0
        if a == "Y": return 1
        return 2  # Z domyślnie

    def _unit_axis(self, axis: str):
        """Jednostkowy wektor osi (X/Y/Z)."""
        i = self._axis_index(axis)
        return (1.0 if i == 0 else 0.0,
                1.0 if i == 1 else 0.0,
                1.0 if i == 2 else 0.0)

    def draw_parallel_probe_points(self, u, v, *, axis="Z", offset=0.6, n=7,
                                   size=22, mirror=False, level=0.0,
                                   marker="o"):
        """
        Rysuje n punktów próbkowania na odcinku u→v, odsuniętych o 'offset'
        wzdłuż osi 'axis' (poziom odniesienia). Opcjonalnie rysuje odbicie
        względem płaszczyzny axis = level (mirror=True).

        Parametry:
          u, v     - końce krawędzi (trójki liczb)
          axis     - 'X'|'Y'|'Z' (oś poziomu odniesienia)
          offset   - skalarny dystans odsunięcia od krawędzi
          n        - liczba próbek (≥2)
          size     - rozmiar markerów
          mirror   - czy rysować lustrzane punkty względem poziomu 'level'
          level    - wartość poziomu (np. Z=0)
          marker   - np. 'o', '^', 's'
        """
        if n < 2:
            n = 2
        # kierunek krawędzi względem osi: kolor 1 dla wzrostu, 0 dla spadku
        idx = self._axis_index(axis)
        dir_sign = 1.0 if (v[idx] - u[idx]) >= 0 else -1.0

        # odsunięcie równoległe do osi poziomu
        ax = self._unit_axis(axis)
        off_vec = (offset * dir_sign * ax[0],
                   offset * dir_sign * ax[1],
                   offset * dir_sign * ax[2])

        # parametryzacja odcinka u→v
        ts = [i / (n - 1) for i in range(n)]
        px, py, pz = [], [], []
        mx, my, mz = [], [], []

        for t in ts:
            p = (u[0] * (1 - t) + v[0] * t,
                 u[1] * (1 - t) + v[1] * t,
                 u[2] * (1 - t) + v[2] * t)
            q = (p[0] + off_vec[0],
                 p[1] + off_vec[1],
                 p[2] + off_vec[2])

            px.append(q[0]);
            py.append(q[1]);
            pz.append(q[2])

            if mirror:
                # lustrzane odbicie względem płaszczyzny 'axis = level'
                # odległość punktu od poziomu wzdłuż tej osi:
                dist = q[idx] - level
                # współrzędna lustrzana:
                q_mirror = list(q)
                q_mirror[idx] = q[idx] - 2.0 * dist
                mx.append(q_mirror[0]);
                my.append(q_mirror[1]);
                mz.append(q_mirror[2])

        # kolor: jasny dla dodatniego kierunku, ciemniejszy dla ujemnego
        alpha_main = 0.95 if dir_sign > 0 else 0.85
        alpha_mirr = 0.55

        # rysuj punkty sondy równoległe do krawędzi
        self.ax.scatter(px, py, pz, s=size, marker=marker, alpha=alpha_main)

        # jeśli mirror=True — dorysuj odbicia
        if mirror and mx:
            self.ax.scatter(mx, my, mz, s=int(size * 0.85), marker=marker, alpha=alpha_mirr)

    def draw_edge_with_probes(self, u, v, kind="S", *,
                              axis="Z", offset=0.6, n=7,
                              edge_lw=2.0, size=22, mirror=False, level=0.0,
                              marker="o"):
        """
        Rysuje krawędź (S/H) oraz równoległe punkty sondy.
        """
        if kind == "S":
            self.draw_edge_S(u, v, lw=edge_lw)
        else:
            self.draw_edge_H(u, v, lw=edge_lw)
        self.draw_parallel_probe_points(u, v, axis=axis, offset=offset, n=n,
                                        size=size, mirror=mirror, level=level,
                                        marker=marker)


# ============================================================
# 4) WEJŚCIE: mapowanie strzałek na S/H
# ============================================================

class InputRouter:
    def __init__(self, widget, move_callback, reset_callback=None, status_callback=None):
        self.widget = widget
        self.move_cb = move_callback
        self.reset_cb = reset_callback or (lambda: None)
        self.status_cb = status_callback or (lambda _msg: None)
        self.h_mode = "X"  # "X" | "Y" | "Z"
        self.bind_all()

    def bind_all(self):
        w = self.widget
        w.bind_all("<Left>", self._left);
        w.bind_all("<Right>", self._right)
        w.bind_all("<Up>", self._up);
        w.bind_all("<Down>", self._down)
        w.bind_all("<Prior>", self._pgup)  # PageUp
        w.bind_all("<Next>", self._pgdn)  # PageDown
        w.bind_all("<space>", self._reset)
        w.bind_all("<F1>", lambda e: self._set_mode("X"))
        w.bind_all("<F2>", lambda e: self._set_mode("Y"))
        w.bind_all("<F3>", lambda e: self._set_mode("Z"))

    def _set_mode(self, m):
        self.h_mode = m
        self.status_cb(f"Tryb H-dominant: {m}")

    def _left(self, e):
        if e.state & 0x0008:
            self._move_H(dx=-1, dy=+1, dz=self._dz())
        else:
            self.move_cb(("S", (-2, 0, 0)))

    def _right(self, e):
        if e.state & 0x0008:
            self._move_H(dx=+1, dy=-1, dz=self._dz())
        else:
            self.move_cb(("S", (+2, 0, 0)))

    def _up(self, e):
        if e.state & 0x0008:
            self._move_H(dx=+1, dy=+1, dz=self._dz(+1))
        else:
            self.move_cb(("S", (0, +2, 0)))

    def _down(self, e):
        if e.state & 0x0008:
            self._move_H(dx=-1, dy=-1, dz=self._dz(-1))
        else:
            self.move_cb(("S", (0, -2, 0)))

    def _pgup(self, _e):
        self.move_cb(("S", (0, 0, +2)))

    def _pgdn(self, _e):
        self.move_cb(("S", (0, 0, -2)))

    def _reset(self, _e):
        self.reset_cb()

    def _dz(self, prefer=0):
        if self.h_mode == "Z": return +1 if prefer >= 0 else -1
        return +1 if self.h_mode == "X" else -1

    def _move_H(self, dx, dy, dz):
        sx = 1 if dx >= 0 else -1
        sy = 1 if dy >= 0 else -1
        sz = 1 if dz >= 0 else -1
        self.move_cb(("H", (sx, sy, sz)))


# ============================================================
# 5) ZAKŁADKI 1–4: MosaicLab+, MetaWorld (TO), Slice2D, Sim
#    (skrócone do sedna, jak w Twojej wersji)
# ============================================================

class _Constructions:
    def __init__(self, origin=(0, 0, 0)):
        self.o = origin;
        self.reset()

    def reset(self):
        self.steps = [];
        self.idx = 0

    def build_triangle(self):
        self.reset();
        u = self.o;
        H1 = (1, 1, 1);
        ex = (2, 0, 0)
        self.steps += [(u, add(u, H1), "H"),
                       (add(u, H1), add(u, ex), "H"),
                       (u, add(u, ex), "S"),
                       (add(u, ex), u, "S")]
        return self

    def build_square(self):
        self.reset();
        u = self.o;
        ex = (2, 0, 0);
        ey = (0, 2, 0)
        self.steps += [(u, add(u, ex), "S"),
                       (add(u, ex), add(u, add(ex, ey)), "S"),
                       (add(u, add(ex, ey)), add(u, ey), "S"),
                       (add(u, ey), u, "S")]
        return self

    def build_five_paths(self):
        self.reset();
        u = self.o;
        ex = (2, 0, 0)
        self.steps.append((u, add(u, ex), "S"))
        for sy in (+1, -1):
            for sz in (+1, -1):
                m = add(u, (1, sy, sz))
                self.steps.append((u, m, "H"))
                self.steps.append((m, add(u, ex), "H"))
        return self


class MosaicLabTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        left = ttk.Frame(self);
        left.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=6)
        right = ttk.Frame(self);
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.scene = Scene3D(right, "Mozaika 14-portowa — prawa, konstrukcje, planowanie")
        self.scene._draw_compass = False;
        self.scene.set_origin((0, 0, 0))
        self.var_compass = tk.BooleanVar(value=False)
        self.var_anim = tk.BooleanVar(value=False)
        panel = ttk.LabelFrame(left, text="Prawa (A*)");
        panel.pack(fill=tk.X, pady=4)
        self.var_wH = tk.DoubleVar(value=1.0);
        self.var_wS = tk.DoubleVar(value=1.0);
        self.var_heur = tk.StringVar(value="mix")
        ttk.Label(panel, text="w_H").grid(row=0, column=0, sticky="e")
        tk.Spinbox(panel, from_=0.1, to=10, increment=0.1, textvariable=self.var_wH, width=6,
                   command=self._render).grid(row=0, column=1, sticky="w")
        ttk.Label(panel, text="w_S").grid(row=0, column=2, sticky="e")
        tk.Spinbox(panel, from_=0.1, to=10, increment=0.1, textvariable=self.var_wS, width=6,
                   command=self._render).grid(row=0, column=3, sticky="w")
        ttk.Label(panel, text="heur.").grid(row=0, column=4, sticky="e")
        ttk.OptionMenu(panel, self.var_heur, "mix", "mix", "l1", "zero", command=lambda _=None: self._render()).grid(
            row=0, column=5, sticky="w")
        cpanel = ttk.LabelFrame(left, text="Konstrukcje");
        cpanel.pack(fill=tk.X, pady=4)
        ttk.Checkbutton(cpanel, text="Animuj kroki", variable=self.var_anim).pack(side=tk.LEFT, padx=8)
        ttk.Button(cpanel, text="Trójkąt H–H–S", command=self._run_triangle).pack(side=tk.LEFT, padx=2)
        ttk.Button(cpanel, text="Kwadrat S4", command=self._run_square).pack(side=tk.LEFT, padx=2)
        ttk.Button(cpanel, text="≥5 ścieżek", command=self._run_five).pack(side=tk.LEFT, padx=2)
        ppanel = ttk.LabelFrame(left, text="Planowanie");
        ppanel.pack(fill=tk.X, pady=4)
        ttk.Label(ppanel, text="Start").pack(side=tk.LEFT)
        self.e_start = tk.Entry(ppanel, width=12);
        self.e_start.insert(0, "0,0,0");
        self.e_start.pack(side=tk.LEFT, padx=4)
        ttk.Label(ppanel, text="Cel").pack(side=tk.LEFT)
        self.e_goal = tk.Entry(ppanel, width=12);
        self.e_goal.insert(0, "6,4,2");
        self.e_goal.pack(side=tk.LEFT, padx=4)
        ttk.Button(ppanel, text="Dopasuj", command=self._snap).pack(side=tk.LEFT, padx=2)
        ttk.Button(ppanel, text="Szukaj", command=self._plan).pack(side=tk.LEFT, padx=2)
        self.lbl = ttk.Label(ppanel, text="");
        self.lbl.pack(side=tk.LEFT, padx=8)
        vpanel = ttk.LabelFrame(left, text="Widok");
        vpanel.pack(fill=tk.X, pady=4)
        ttk.Checkbutton(vpanel, text="Kompas 14", variable=self.var_compass, command=self._render).pack(side=tk.LEFT)
        ttk.Button(vpanel, text="Reset wid.", command=self.scene.reset_view).pack(side=tk.LEFT, padx=2)
        ttk.Button(vpanel, text="Zapisz PNG", command=self._save_png).pack(side=tk.LEFT, padx=2)
        self.con = _Constructions();
        self.after(0, self._render)

    def _laws(self):
        return Laws(self.var_wH.get(), self.var_wS.get(), self.var_heur.get())

    def _render(self):
        self.scene.clear("Mozaika 14-portowa — prawa, konstrukcje, planowanie")
        self.scene._draw_compass = bool(self.var_compass.get())
        if self.scene._draw_compass: self.scene.draw_compass()
        self.scene.canvas.draw_idle()

    def _run_triangle(self):
        self.con.build_triangle();
        self._play()

    def _run_square(self):
        self.con.build_square();
        self._play()

    def _run_five(self):
        self.con.build_five_paths();
        self._play()

    def _play(self):
        self._render();
        steps = list(self.con.steps)
        xs, ys, zs = [], [], []
        for (u, v, t) in steps:
            (self.scene.draw_edge_S if t == "S" else self.scene.draw_edge_H)(u, v, lw=2.2)
            xs += [u[0], v[0]];
            ys += [u[1], v[1]];
            zs += [u[2], v[2]]
        self.scene.set_limits(xs or [0], ys or [0], zs or [0]);
        self.scene.canvas.draw_idle()

    def _snap(self):
        try:
            s = parse_point_text(self.e_start.get());
            g = parse_point_text(self.e_goal.get())
        except ValueError:
            self.lbl.config(text="Najpierw wprowadź x,y,z.");
            return
        s2 = project_to_l2(s);
        g2 = project_to_l2(g)
        self.e_start.delete(0, "end");
        self.e_start.insert(0, f"{s2[0]},{s2[1]},{s2[2]}")
        self.e_goal.delete(0, "end");
        self.e_goal.insert(0, f"{g2[0]},{g2[1]},{g2[2]}")
        self.lbl.config(text="Dopasowano do x≡y≡z (mod 2).")

    def _plan(self):
        try:
            start = parse_point_text(self.e_start.get());
            goal = parse_point_text(self.e_goal.get())
        except ValueError as e:
            self.lbl.config(text=f"Błąd: {e}");
            return
        if not parity_ok(start) or not parity_ok(goal):
            self.lbl.config(text="Punkty muszą spełniać x≡y≡z (mod 2).");
            return
        path, cost, seen = astar(start, goal, self._laws())
        self._render()
        if not path:
            self.lbl.config(text=f"Brak ścieżki (visited={seen}).")
            return

        ok, msg = validate_path(path)
        xs, ys, zs = [], [], []
        for u, v in zip(path, path[1:]):
            (self.scene.draw_edge_S if edge_type(u, v) == "S" else self.scene.draw_edge_H)(u, v, lw=2.0)
            xs += [u[0], v[0]];
            ys += [u[1], v[1]];
            zs += [u[2], v[2]]
        self.scene.set_limits(xs, ys, zs)
        self.scene.canvas.draw_idle()

        prefix = "✔" if ok else "⚠"
        self.lbl.config(text=f"{prefix} {msg} | koszt={cost:.3f}, kroki={len(path) - 1}, odwiedzonych={seen}")

    def _save_png(self):
        file = filedialog.asksaveasfilename(title="Zapisz obraz", defaultextension=".png",
                                            filetypes=[("PNG", "*.png")], initialfile="mosaic_lab.png")
        if file: self.scene.fig.savefig(file, dpi=150)


class MetaWorldTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        left = ttk.Frame(self);
        left.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=6)
        right = ttk.Frame(self);
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.scene = Scene3D(right, "Meta-świat — siatka TO (wireframe)")
        self.scene._draw_compass = False;
        self.scene.set_origin((0, 0, 0))
        self.var_centers = tk.BooleanVar(value=False)
        self.var_edgesS = tk.BooleanVar(value=False)
        self.var_edgesH = tk.BooleanVar(value=False)
        self.var_squares = tk.BooleanVar(value=False)
        self.var_hexes = tk.BooleanVar(value=False)
        opts = ttk.LabelFrame(left, text="Warstwy");
        opts.pack(fill=tk.X, pady=4)
        ttk.Checkbutton(opts, text="Centra (BCC)", variable=self.var_centers, command=self.render).grid(row=0, column=0,
                                                                                                        sticky="w")
        ttk.Checkbutton(opts, text="Krawędzie S", variable=self.var_edgesS, command=self.render).grid(row=0, column=1,
                                                                                                      sticky="w")
        ttk.Checkbutton(opts, text="Krawędzie H", variable=self.var_edgesH, command=self.render).grid(row=0, column=2,
                                                                                                      sticky="w")
        ttk.Checkbutton(opts, text="Kwadraty ⟂ S", variable=self.var_squares, command=self.render).grid(row=1, column=0,
                                                                                                        sticky="w")
        ttk.Checkbutton(opts, text="Heksy ⟂ H", variable=self.var_hexes, command=self.render).grid(row=1, column=1,
                                                                                                   sticky="w")
        grid = ttk.LabelFrame(left, text="Parametry siatki");
        grid.pack(fill=tk.X, pady=4)
        ttk.Label(grid, text="Zasięg R").grid(row=0, column=0, sticky="e")
        self.spin_R = tk.Spinbox(grid, from_=1, to=5, width=5, command=self.render);
        self.spin_R.delete(0, "end");
        self.spin_R.insert(0, "2");
        self.spin_R.grid(row=0, column=1, sticky="w")
        ttk.Checkbutton(grid, text="Kompas 14", command=self._toggle_compass).grid(row=0, column=2, sticky="w")
        ttk.Button(grid, text="Reset widoku", command=self.scene.reset_view).grid(row=0, column=3, padx=4)
        ttk.Label(left, text="Włącz selektywnie warstwy. Domyślnie scena pusta.", wraplength=360, justify="left").pack(
            fill=tk.X, pady=6)
        self.after(0, self.render)

    def _toggle_compass(self):
        self.scene._draw_compass = not self.scene._draw_compass;
        self.render()

    def _centers(self, R):
        pts = []
        for x in range(-2 * R, 2 * R + 1):
            for y in range(-2 * R, 2 * R + 1):
                for z in range(-2 * R, 2 * R + 1):
                    p = (x, y, z)
                    if parity_ok(p): pts.append(p)
        return set(pts)

    def render(self):
        self.scene.clear("Meta-świat — siatka TO (wireframe)")
        if self.scene._draw_compass: self.scene.draw_compass()
        R = int(self.spin_R.get());
        centers = self._centers(R)
        xs, ys, zs = [], [], []
        if self.var_centers.get():
            pts = list(centers);
            xs = [p[0] for p in pts];
            ys = [p[1] for p in pts];
            zs = [p[2] for p in pts]
            self.scene.draw_points(pts, size=8)
        if self.var_edgesS.get():
            for c in centers:
                for s in S_VECS:
                    v = add(c, s)
                    if v in centers: self.scene.draw_edge_S(c, v, lw=1.0)
        if self.var_edgesH.get():
            for c in centers:
                for h in H_VECS:
                    v = add(c, h)
                    if v in centers: self.scene.draw_edge_H(c, v, lw=0.9)
        # proste kontury twarzy dla intuicji
        if self.var_squares.get():
            for c in centers:
                for axis in ('x', 'y', 'z'):
                    cx, cy, cz = c
                    if axis == 'x':
                        poly = [(cx, cy - 1, cz - 1), (cx, cy + 1, cz - 1), (cx, cy + 1, cz + 1), (cx, cy - 1, cz + 1),
                                (cx, cy - 1, cz - 1)]
                    elif axis == 'y':
                        poly = [(cx - 1, cy, cz - 1), (cx + 1, cy, cz - 1), (cx + 1, cy, cz + 1), (cx - 1, cy, cz + 1),
                                (cx - 1, cy, cz - 1)]
                    else:
                        poly = [(cx - 1, cy - 1, cz), (cx + 1, cy - 1, cz), (cx + 1, cy + 1, cz), (cx - 1, cy + 1, cz),
                                (cx - 1, cy - 1, cz)]
                    self.scene.ax.plot([p[0] for p in poly], [p[1] for p in poly], [p[2] for p in poly], linewidth=1.0)
        if self.var_hexes.get():
            try:
                import numpy as np
                for c in centers:
                    for h in H_VECS:
                        cx, cy, cz = c
                        d = (float(h[0]), float(h[1]), float(h[2]))
                        dn = np.array(d);
                        dn = dn / np.linalg.norm(dn)
                        # wybór wektora niekolinearnie z dn
                        tmp = np.array([1.0, 0.0, 0.0])
                        if abs(float(dn[0])) > 0.8:
                            tmp = np.array([0.0, 1.0, 0.0])
                        u = tmp - np.dot(tmp, dn) * dn
                        u = u / np.linalg.norm(u)
                        v = np.cross(dn, u)
                        R = math.sqrt(2)
                        verts = []
                        for k in range(7):
                            ang = 2 * math.pi * k / 6.0
                            p = (cx + R * (math.cos(ang) * u[0] + math.sin(ang) * v[0]),
                                 cy + R * (math.cos(ang) * u[1] + math.sin(ang) * v[1]),
                                 cz + R * (math.cos(ang) * u[2] + math.sin(ang) * v[2]))
                            verts.append(p)
                        self.scene.ax.plot([p[0] for p in verts], [p[1] for p in verts], [p[2] for p in verts],
                                           linewidth=0.9, alpha=0.85)
            except Exception:
                # Brak numpy lub środowisko bez obsługi — pomiń warstwę heksów
                pass
        self.scene.set_limits(xs or [0], ys or [0], zs or [0]);
        self.scene.canvas.draw_idle()


class Slice2DTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        left = ttk.Frame(self);
        left.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=6)
        right = ttk.Frame(self);
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.scene = Scene3D(right, "Przekroje 2D — kwadraty S i rzuty H")
        self.scene._draw_compass = True;
        self.scene.set_origin((0, 0, 0))
        pane = ttk.LabelFrame(left, text="Parametry");
        pane.pack(fill=tk.X, pady=4)
        ttk.Label(pane, text="Płaszczyzna").grid(row=0, column=0, sticky="e")
        self.var_plane = tk.StringVar(value="XY")
        ttk.OptionMenu(pane, self.var_plane, "XY", "XY", "YZ", "ZX", command=lambda _=None: self.render()).grid(row=0,
                                                                                                                column=1,
                                                                                                                sticky="w")
        ttk.Label(pane, text="Poziom").grid(row=0, column=2, sticky="e")
        self.spin_level = tk.Spinbox(pane, from_=-10, to=10, width=6, command=self.render);
        self.spin_level.delete(0, "end");
        self.spin_level.insert(0, "0");
        self.spin_level.grid(row=0, column=3, sticky="w")
        ttk.Label(pane, text="Zasięg R").grid(row=1, column=0, sticky="e")
        self.spin_R = tk.Spinbox(pane, from_=2, to=15, width=6, command=self.render);
        self.spin_R.delete(0, "end");
        self.spin_R.insert(0, "8");
        self.spin_R.grid(row=1, column=1, sticky="w")
        self.var_showS = tk.BooleanVar(value=True);
        self.var_showH = tk.BooleanVar(value=True)
        ttk.Checkbutton(pane, text="Pokaż S", variable=self.var_showS, command=self.render).grid(row=1, column=2,
                                                                                                 sticky="w")
        ttk.Checkbutton(pane, text="Pokaż H", variable=self.var_showH, command=self.render).grid(row=1, column=3,
                                                                                                 sticky="w")
        self.after(0, self.render)

    def render(self):
        self.scene.clear("Przekroje 2D — kwadraty S i rzuty H")
        if self.scene._draw_compass: self.scene.draw_compass()
        plane = self.var_plane.get();
        level = int(self.spin_level.get());
        R = int(self.spin_R.get())
        pts = []
        for x in range(-R, R + 1):
            for y in range(-R, R + 1):
                for z in range(-R, R + 1):
                    if not parity_ok((x, y, z)): continue
                    if (plane == "XY" and z != level) or (plane == "YZ" and x != level) or (
                            plane == "ZX" and y != level): continue
                    pts.append((x, y, z))
        xs, ys, zs = [], [], []
        for (x, y, z) in pts:
            if self.var_showS.get():
                if plane == "XY":
                    poly = [(x - 1, y - 1, level), (x + 1, y - 1, level), (x + 1, y + 1, level), (x - 1, y + 1, level),
                            (x - 1, y - 1, level)]
                elif plane == "YZ":
                    poly = [(level, y - 1, z - 1), (level, y + 1, z - 1), (level, y + 1, z + 1), (level, y - 1, z + 1),
                            (level, y - 1, z - 1)]
                else:
                    poly = [(z - 1, level, x - 1), (z + 1, level, x - 1), (z + 1, level, x + 1), (z - 1, level, x + 1),
                            (z - 1, level, x - 1)]
                self.scene.ax.plot([q[0] for q in poly], [q[1] for q in poly], [q[2] for q in poly], linewidth=0.9)
            if self.var_showH.get():
                if plane == "XY":
                    self.scene.ax.plot([x - 1, x + 1], [y - 1, y + 1], [level, level], linewidth=0.8, alpha=0.7)
                    self.scene.ax.plot([x - 1, x + 1], [y + 1, y - 1], [level, level], linewidth=0.8, alpha=0.7)
                elif plane == "YZ":
                    self.scene.ax.plot([level, level], [y - 1, y + 1], [z - 1, z + 1], linewidth=0.8, alpha=0.7)
                    self.scene.ax.plot([level, level], [y - 1, y + 1], [z + 1, z - 1], linewidth=0.8, alpha=0.7)
                else:
                    self.scene.ax.plot([z - 1, z + 1], [level, level], [x - 1, x + 1], linewidth=0.8, alpha=0.7)
                    self.scene.ax.plot([z - 1, z + 1], [level, level], [x + 1, x - 1], linewidth=0.8, alpha=0.7)
            xs.append(x);
            ys.append(y);
            zs.append(z)
        self.scene.set_limits(xs or [0], ys or [0], zs or [0]);
        self.scene.canvas.draw_idle()


class SimTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        left = ttk.Frame(self);
        left.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=6)
        right = ttk.Frame(self);
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.scene = Scene3D(right, "Symulacje — budowa artefaktów S/H z klawiatury")
        self.scene._draw_compass = True;
        self.scene.set_origin((0, 0, 0))
        self.cursor = (0, 0, 0);
        self.path = []
        ttk.Checkbutton(left, text="Kompas 14", command=self._toggle_compass).pack(anchor="w")
        ttk.Button(left, text="Reset widoku", command=self.scene.reset_view).pack(anchor="w", pady=2)
        ttk.Button(left, text="Zapisz PNG", command=self._save_png).pack(anchor="w", pady=2)
        ttk.Label(left, text=("Sterowanie: strzałki = S, PgUp/PgDn = Z±, ALT+strzałki = H, "
                              "F1/F2/F3: tryb H-dominant (X/Y/Z), Spacja: reset."), wraplength=360,
                  justify="left").pack(fill=tk.X, pady=6)
        self.router = InputRouter(self, self._on_move, self._on_reset)
        self._repaint()

    def _toggle_compass(self):
        self.scene._draw_compass = not self.scene._draw_compass;
        self._repaint()

    def _save_png(self):
        file = filedialog.asksaveasfilename(title="Zapisz obraz", defaultextension=".png",
                                            filetypes=[("PNG", "*.png")], initialfile="sim_scene.png")
        if file: self.scene.fig.savefig(file, dpi=150)

    def _on_reset(self):
        self.cursor = (0, 0, 0);
        self.path.clear();
        self._repaint()

    def _on_move(self, step):
        kind, vec = step;
        u = self.cursor;
        v = add(u, vec)
        if not parity_ok(v): v = project_to_l2(v)
        self.path.append((u, v, kind));
        self.cursor = v;
        self._repaint()

    def _repaint(self):
        self.scene.clear("Symulacje — budowa artefaktów S/H z klawiatury")
        if self.scene._draw_compass: self.scene.draw_compass()
        xs, ys, zs = [0], [0], [0]
        for (u, v, k) in self.path:
            (self.scene.draw_edge_S if k == "S" else self.scene.draw_edge_H)(u, v, lw=2.0)
            xs += [u[0], v[0]];
            ys += [u[1], v[1]];
            zs += [u[2], v[2]]
        self.scene.draw_points([self.cursor], size=30)
        self.scene.set_limits(xs, ys, zs);
        self.scene.canvas.draw_idle()


# ============================================================
# 6) AST LAB — integracja honey_ast_hub jako zakładki
# ============================================================

class AstGraph:
    """Minimalny graf AST:
       - S: parent→child i sibling→next,
       - H: Use→Def oraz arg→Function (powiązanie ramki).
    """

    def __init__(self):
        self.nodes = []  # [(id, label)]
        self.edges = []  # [(u,v,type)]  type in {"S","H"}
        self._id = 0
        self.idmap = {}  # ast.AST -> id
        self.bindings = {}  # name -> def_id

    def nid(self, node, label=None):
        if node in self.idmap: return self.idmap[node]
        self._id += 1
        lbl = label or type(node).__name__
        self.nodes.append((self._id, lbl))
        self.idmap[node] = self._id
        return self._id

    def addS(self, u, v):
        self.edges.append((u, v, "S"))

    def addH(self, u, v):
        self.edges.append((u, v, "H"))

    def build_from_ast(self, node):
        def walk(parent, n):
            pid = self.nid(n)
            if parent is not None: self.addS(parent, pid)
            prev = None
            for _field, value in ast.iter_fields(n):
                if isinstance(value, ast.AST):
                    cid = self.nid(value);
                    self.addS(pid, cid)
                    if prev is not None: self.addS(prev, cid)
                    walk(pid, value);
                    prev = cid
                elif isinstance(value, list):
                    for it in value:
                        if isinstance(it, ast.AST):
                            cid = self.nid(it);
                            self.addS(pid, cid)
                            if prev is not None: self.addS(prev, cid)
                            walk(pid, it);
                            prev = cid

        walk(None, node)
        for n in ast.walk(node):
            if isinstance(n, ast.FunctionDef):
                f_id = self.nid(n)
                for a in n.args.args:
                    a_id = self.nid(a);
                    self.bindings[a.arg] = a_id;
                    self.addH(a_id, f_id)
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        self.bindings[t.id] = self.nid(t)
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                use_id = self.nid(n);
                def_id = self.bindings.get(n.id)
                if def_id: self.addH(use_id, def_id)
        return self


def ast_to_scene_coords(tree):
    depth = {};
    children_map = {}

    def walk_levels(n, d=0):
        depth[n] = d
        children = []
        for _f, val in ast.iter_fields(n):
            if isinstance(val, ast.AST):
                children.append(val)
            elif isinstance(val, list):
                children.extend([it for it in val if isinstance(it, ast.AST)])
        children_map[n] = children
        for ch in children: walk_levels(ch, d + 1)

    walk_levels(tree)
    per_level = {};
    [per_level.setdefault(d, []).append(n) for n, d in depth.items()]
    order_on_level = {}
    for d, nodes in per_level.items():
        for i, n in enumerate(nodes): order_on_level[n] = i
    buckets = {};
    type_bucket = {}
    for n in depth.keys():
        t = type(n).__name__
        if t not in buckets: buckets[t] = len(buckets)
        type_bucket[n] = buckets[t]
    coords = {}
    for n in depth.keys():
        x = 2 * order_on_level[n];
        y = 2 * type_bucket[n];
        z = 2 * depth[n]
        coords[n] = (x, y, z)
    labels = {n: type(n).__name__ for n in depth.keys()}
    return coords, labels, per_level


def compute_meta_layers(tree, G, coords, per_level, *, mode="depth-bucket", bucket=2):
    """
    Zwraca: groups: list of dict:
        {"name": str, "nodes": [ast.AST], "center": (x,y,z), "bbox": (xmin,xmax,ymin,ymax,zmin,zmax)}
    oraz mapowanie node->center (anchor) i node->group_name.
    """
    # przygotuj pomocnicze słowniki
    id2node = {i: n for (n, i) in G.idmap.items()}
    node2id = {n: i for (i, n) in id2node.items()}

    groups = []
    node2group = {}
    group2nodes = {}

    if mode == "type":
        # grupy po nazwie typu
        for n in coords.keys():
            key = type(n).__name__
            group2nodes.setdefault(key, []).append(n)
    elif mode == "defuse":
        # komponenty spójności po krawędziach H (Use<->Def, arg->Function)
        dsu = DSU()
        for (u_id, v_id, t) in G.edges:
            if t == "H":
                dsu.union(u_id, v_id)
        comp = {}
        for n in coords.keys():
            root = dsu.find(node2id[n])
            comp.setdefault(root, []).append(n)
        # nazwijmy grupy po indeksie korzenia
        for root, ns in comp.items():
            key = f"DU#{root}"
            group2nodes[key] = ns
    else:
        # depth-bucket (domyślnie)
        # per_level: dict depth -> [nodes]; zrzucamy do kubełków co 'bucket'
        for d, nodes in per_level.items():
            key = f"D{d // bucket}"
            group2nodes.setdefault(key, []).extend(nodes)

    # policz centra i bboxy
    centers = {}
    for key, ns in group2nodes.items():
        pts = [coords[n] for n in ns if n in coords]
        if not pts:
            continue
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        cz = sum(p[2] for p in pts) / len(pts)
        bb = bbox_of(pts)
        groups.append({"name": key, "nodes": ns, "center": (cx, cy, cz), "bbox": bb})
        for n in ns:
            node2group[n] = key
            centers[n] = (cx, cy, cz)

    return groups, centers, node2group

# ============================================================
# A) EWOLUCJA TOPOLOGICZNA MOZAIKI (DARPA "mosaic/honeycomb")
#     - supergraf grup (węzeł=grupa, krawędź=agregat S/H)
#     - wybór reprezentantów grup vs. pełny detal
#     - modulacja grubości krawędzi względem λ
#     - stabilny aspekt pudełka (bez "fałszywego zoomu")
# ============================================================

def build_supergraph(groups, G, node2group):
    """
    Buduje supergraf:
      - super_nodes: [(group_name, center(x,y,z), size)]
      - super_edges: [(gA, gB, kind, count)], kind ∈ {"S","H"}, count = liczba krawędzi oryginalnych między grupami.

    Założenie "honeycomb": łączność międzykomórkowa (H/S) jest agregowana;
    miara siły połączenia = liczba realnych połączeń (count).
    """
    centers = {g["name"]: g["center"] for g in groups}
    sizes   = {g["name"]: len(g["nodes"]) for g in groups}

    id2node = {i: n for (n, i) in G.idmap.items()}
    cntS, cntH = {}, {}

    for (u_id, v_id, t) in G.edges:
        u = id2node[u_id]; v = id2node[v_id]
        gu = node2group.get(u); gv = node2group.get(v)
        if not gu or not gv or gu == gv:
            continue
        key = tuple(sorted((gu, gv)))
        if t == "S":
            cntS[key] = cntS.get(key, 0) + 1
        else:
            cntH[key] = cntH.get(key, 0) + 1

    super_nodes = [(name, centers[name], sizes[name]) for name in centers.keys()]
    super_edges = []
    for (ga, gb), c in cntS.items():
        super_edges.append((ga, gb, "S", c))
    for (ga, gb), c in cntH.items():
        super_edges.append((ga, gb, "H", c))
    return super_nodes, super_edges


def select_nodes_for_lambda(per_level, coords_plot, zmax, limit, groups, lam, threshold=0.6, evolve=True):
    """
    Daje zbiór widocznych węzłów dla danego poziomu abstrakcji λ.
      - dla małej λ lub gdy evolve=False: detal (do 'limit' per poziom z filtrem Z)
      - dla dużej λ (λ >= threshold): 1 reprezentant na grupę (najbliższy centroidowi)
    """
    if not evolve or lam < threshold:
        keep = set()
        for _d, nodes_on_level in per_level.items():
            nodes_sorted = sorted(nodes_on_level, key=lambda n: coords_plot[n][0])
            for i, n in enumerate(nodes_sorted):
                x, y, z = coords_plot[n]
                if z <= 2 * zmax and i < limit:
                    keep.add(n)
        return keep

    # λ wysokie: kontrakcja do reprezentantów grup
    import math as _m
    keep = set()
    for g in groups:
        cx, cy, cz = g["center"]
        def _d2(n):
            x, y, z = coords_plot[n]; dx, dy, dz = (x-cx, y-cy, z-cz)
            return dx*dx + dy*dy + dz*dz
        if g["nodes"]:
            rep = min(g["nodes"], key=_d2)
            keep.add(rep)
    return keep


def edge_style_for_lambda(gu, gv, *, lam, threshold, lw_base, evolve):
    """
    Reguła „mosaic”: intra-group zanika wraz z λ, inter-group rośnie subtelnie.
    Zwraca: linewidth (float) lub None gdy krawędź ma być wyłączona.
    """
    same = (gu is not None and gv is not None and gu == gv)
    if not evolve:
        return max(0.4, lw_base * 0.6)
    if same:
        # wewnątrz komórki (plastra) wygaszamy; po progu usuwamy
        if lam >= threshold:
            return None
        return max(0.2, lw_base * (1.0 - lam))
    # między komórkami — podkreślamy umiarkowanie wraz z λ
    return max(0.4, lw_base * (0.6 + 0.6 * lam))


def draw_supergraph_layer(ax, groups, super_edges, lam, *, emphasize_H=0.8, emphasize_S=1.0):
    """
    Rysuje warstwę supergrafu między centroidami grup (tylko dla dużych λ).
    Grubość ∝ liczbie połączeń i λ. Węzły: markery '^' z rozmiarem ~|C|.
    """
    name2center = {g["name"]: g["center"] for g in groups}
    name2size   = {g["name"]: len(g["nodes"]) for g in groups}

    # super-krawędzie
    for (ga, gb, kind, count) in super_edges:
        ca = name2center[ga]; cb = name2center[gb]
        scale = (emphasize_H if kind == "H" else emphasize_S)
        lw = scale * (1.0 + 0.15 * count) * (0.4 + 0.6 * lam)
        # rysunek jako segment prosty (w AST-Lab oba typy są liniami)
        ax.plot([ca[0], cb[0]], [ca[1], cb[1]], [ca[2], cb[2]], linewidth=lw)

    # super-węzły
    for name, (cx, cy, cz) in name2center.items():
        size = name2size.get(name, 1)
        ax.scatter([cx], [cy], [cz], s=40 + 6 * size * lam, marker="^")
        ax.text(cx, cy, cz + 0.4, name, fontsize=7)


def draw_edges_evolving(scene, G, id2node, node2group, coords_plot, nodes_visible,
                        lam, threshold, lwS, lwH, show_S, show_H, sel_S, sel_H, groups):
    """
    Jednolity renderer krawędzi S/H z modulacją λ i wsparciem selektorów.
    Zwraca: (xs, ys, zs) punktów (do ustawiania limitów osi).
    """
    ax = scene.ax
    xs, ys, zs = [], [], []

    def _record(n):
        x, y, z = coords_plot[n]; xs.append(x); ys.append(y); zs.append(z)

    for n in nodes_visible:
        _record(n)

    if show_S:
        for (u_id, v_id, t) in G.edges:
            if t != "S":
                continue
            u = id2node[u_id]; v = id2node[v_id]
            if u not in nodes_visible or v not in nodes_visible:
                continue
            gu = node2group.get(u); gv = node2group.get(v)
            lw = edge_style_for_lambda(gu, gv, lam=lam, threshold=threshold, lw_base=lwS, evolve=True)
            if lw is None:
                continue
            (x1, y1, z1) = coords_plot[u]; (x2, y2, z2) = coords_plot[v]
            scene.draw_edge_S((x1, y1, z1), (x2, y2, z2), lw=lw)

        if sel_S:
            for (u_id, v_id, t) in G.edges:
                if t != "S":
                    continue
                u = id2node[u_id]; v = id2node[v_id]
                if u not in nodes_visible or v not in nodes_visible:
                    continue
                (x1, y1, z1) = coords_plot[u]; (x2, y2, z2) = coords_plot[v]
                tags = set()
                if x2 != x1: tags.add("X+" if x2 > x1 else "X-")
                if y2 != y1: tags.add("Y+" if y2 > y1 else "Y-")
                if z2 != z1: tags.add("Z+" if z2 > z1 else "Z-")
                if tags & sel_S:
                    scene.draw_edge_S((x1, y1, z1), (x2, y2, z2), lw=lwS)

    if show_H:
        for (u_id, v_id, t) in G.edges:
            if t != "H":
                continue
            u = id2node[u_id]; v = id2node[v_id]
            if u not in nodes_visible or v not in nodes_visible:
                continue
            gu = node2group.get(u); gv = node2group.get(v)
            lw = edge_style_for_lambda(gu, gv, lam=lam, threshold=threshold, lw_base=lwH, evolve=True)
            if lw is None:
                continue
            (x1, y1, z1) = coords_plot[u]; (x2, y2, z2) = coords_plot[v]
            scene.draw_edge_H((x1, y1, z1), (x2, y2, z2), lw=lw)

        if sel_H:
            for (u_id, v_id, t) in G.edges:
                if t != "H":
                    continue
                u = id2node[u_id]; v = id2node[v_id]
                if u not in nodes_visible or v not in nodes_visible:
                    continue
                (x1, y1, z1) = coords_plot[u]; (x2, y2, z2) = coords_plot[v]
                lab = ("+" if x2 > x1 else "-") + ("+" if y2 > y1 else "-") + ("+" if z2 > z1 else "-")
                if lab in sel_H:
                    scene.draw_edge_H((x1, y1, z1), (x2, y2, z2), lw=lwH)

    return xs, ys, zs


def set_box_aspect_equal(ax):
    """Stabilna ramka 3D — bez efektu pseudo-zoomu przy zmianie λ."""
    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        # fallback: wyrównaj limity do tej samej rozpiętości
        xlim = ax.get_xlim3d(); ylim = ax.get_ylim3d(); zlim = ax.get_zlim3d()
        spans = (xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0])
        maxs = max(spans)
        def _centered(lim):
            c = 0.5 * (lim[0] + lim[1])
            return (c - 0.5*maxs, c + 0.5*maxs)
        ax.set_xlim3d(*_centered(xlim))
        ax.set_ylim3d(*_centered(ylim))
        ax.set_zlim3d(*_centered(zlim))


def astlab_install_topology_controls(astlab, meta_frame):
    """
    Dodaje kontrolki do AST-Lab: „Ewolucja topologii” i próg intra-grup (λ*).
    Użycie: wywołaj raz w AstLabTab.__init__: astlab_install_topology_controls(self, meta)
    """
    astlab.var_evolve = tk.BooleanVar(value=True)
    ttk.Checkbutton(meta_frame, text="Ewolucja topologii", variable=astlab.var_evolve,
                    command=lambda: astlab.render()).grid(row=3, column=0, sticky="w")

    ttk.Label(meta_frame, text="próg intra (λ*)").grid(row=3, column=1, sticky="e")
    astlab.scale_thresh = tk.Scale(meta_frame, from_=0.0, to=1.0, resolution=0.05,
                                   orient=tk.HORIZONTAL, length=120,
                                   command=lambda _v: astlab.render())
    astlab.scale_thresh.set(0.6)
    astlab.scale_thresh.grid(row=3, column=2, columnspan=2, sticky="ew")


# (opcjonalnie) Test heurystyki A* — dowód empiryczny admissibility/consistency
def empirical_heuristic_report(laws, R=4, trials=64, seed=0):
    """
    Losowo próbuje par start/goal w kratownicy parzystości (x≡y≡z mod 2),
    sprawdza czy h(s,g) ≤ koszt najkrótszy. Zwraca statystyki.
    Nie używane w GUI; do jednostkowych testów jakości naukowej.
    """
    import random
    random.seed(seed)

    def _rand_point():
        # generuj do skutku punkt z parytetem
        while True:
            p = (random.randint(-2*R, 2*R),
                 random.randint(-2*R, 2*R),
                 random.randint(-2*R, 2*R))
            if parity_ok(p):
                return p

    violations = 0
    deltas = []
    for _ in range(trials):
        s = _rand_point(); g = _rand_point()
        h = laws.h_estimate(s, g)
        path, cost, _seen = astar(s, g, laws, max_nodes=100000)
        if path is None:
            continue
        deltas.append(cost - h)
        if h - cost > 1e-9:
            violations += 1
    return {
        "trials": trials,
        "violations": violations,
        "min_margin": min(deltas) if deltas else None,
        "avg_margin": sum(deltas)/len(deltas) if deltas else None,
        "max_margin": max(deltas) if deltas else None,
    }

DEFAULT_SNIPPET = """\
def f(x):
    y = x
    if y > 0:
        y = y - 1
    return y + 1
"""


class AstLabTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        left = ttk.Frame(self);
        left.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)
        right = ttk.Frame(self);
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        ttk.Label(left, text="Kod Pythona (Ctrl+Enter = render):").pack(anchor="w")
        self.txt = tk.Text(left, width=60, height=32, font=("Consolas", 10))
        self.txt.pack(fill=tk.Y);
        self.txt.insert("1.0", DEFAULT_SNIPPET)

        ctrl = ttk.LabelFrame(left, text="Ustawienia")
        ctrl.pack(fill=tk.X, pady=8)
        self.var_show_S = tk.BooleanVar(value=True)
        self.var_show_H = tk.BooleanVar(value=True)
        self.var_show_labels = tk.BooleanVar(value=True)
        self.var_auto_render = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="Pokaż S", variable=self.var_show_S, command=lambda: self.render()).grid(row=0,
                                                                                                            column=0,
                                                                                                            sticky="w")
        ttk.Checkbutton(ctrl, text="Pokaż H", variable=self.var_show_H, command=lambda: self.render()).grid(row=0,
                                                                                                            column=1,
                                                                                                            sticky="w")
        ttk.Checkbutton(ctrl, text="Etykiety", variable=self.var_show_labels, command=lambda: self.render()).grid(row=0,
                                                                                                                  column=2,
                                                                                                                  sticky="w")
        ttk.Checkbutton(ctrl, text="Auto-render (0.6 s)", variable=self.var_auto_render).grid(row=0, column=3,
                                                                                              sticky="w")
        ttk.Label(ctrl, text="Punkty").grid(row=1, column=0, sticky="w")
        self.scale_pts = tk.Scale(ctrl, from_=2, to=20, orient=tk.HORIZONTAL, command=lambda _v: self.render())
        self.scale_pts.set(8);
        self.scale_pts.grid(row=1, column=1, sticky="ew")
        ttk.Label(ctrl, text="Grubość S").grid(row=1, column=2, sticky="w")
        self.scale_ls = tk.Scale(ctrl, from_=0.5, to=4, resolution=0.1, orient=tk.HORIZONTAL,
                                 command=lambda _v: self.render())
        self.scale_ls.set(1.5);
        self.scale_ls.grid(row=1, column=3, sticky="ew")
        ttk.Label(ctrl, text="Grubość H").grid(row=2, column=0, sticky="w")
        self.scale_lh = tk.Scale(ctrl, from_=0.5, to=4, resolution=0.1, orient=tk.HORIZONTAL,
                                 command=lambda _v: self.render())
        self.scale_lh.set(1.0);
        self.scale_lh.grid(row=2, column=1, sticky="ew")
        ttk.Label(ctrl, text="Limit/poziom").grid(row=2, column=2, sticky="w")
        self.spin_limit = tk.Spinbox(ctrl, from_=1, to=999, width=7, command=lambda: self.render())
        self.spin_limit.delete(0, "end");
        self.spin_limit.insert(0, "999");
        self.spin_limit.grid(row=2, column=3, sticky="w")
        ttk.Label(ctrl, text="Max Z").grid(row=3, column=0, sticky="w")
        self.spin_depth = tk.Spinbox(ctrl, from_=0, to=999, width=7, command=lambda: self.render())
        self.spin_depth.delete(0, "end");
        self.spin_depth.insert(0, "999");
        self.spin_depth.grid(row=3, column=1, sticky="w")

        # pad kierunków (S + H)
        pad = ttk.LabelFrame(left, text="Pad kierunków");
        pad.pack(fill=tk.X, pady=8)
        self.sel_S = set();
        self.sel_H = set()
        srow = ttk.Frame(pad);
        srow.pack(fill=tk.X)
        ttk.Label(srow, text="S:").pack(side=tk.LEFT)
        for lab in ("X+", "X-", "Y+", "Y-", "Z+", "Z-"):
            ttk.Button(srow, text=lab, width=4, command=lambda L=lab: self._toggle_S(L)).pack(side=tk.LEFT, padx=2)
        hrow = ttk.Frame(pad);
        hrow.pack(fill=tk.X)
        ttk.Label(hrow, text="H:").pack(side=tk.LEFT)
        for lab in ("+++", "++-", "+-+", "+--", "-++", "-+-", "--+", "---"):
            ttk.Button(hrow, text=lab, width=4, command=lambda L=lab: self._toggle_H(L)).pack(side=tk.LEFT, padx=2)
        clrow = ttk.Frame(pad);
        clrow.pack(fill=tk.X)
        ttk.Button(clrow, text="Wyczyść", command=self._clear_dirs).pack(side=tk.LEFT, padx=2)

        # scena 3D
        self.scene = Scene3D(right, "AST → Kompas 3D")
        self.scene._draw_compass = True;
        self.scene.set_origin((0, 0, 0))

        # status + skróty
        self.status = tk.StringVar(value="Gotowe")
        ttk.Label(right, textvariable=self.status, anchor="w").pack(fill=tk.X)
        self._render_job = None
        self.txt.bind("<KeyRelease>", self._on_key)
        self.bind_all("<Control-Return>", lambda e: self.render())

        # === META PANEL (po prawej) ===
        meta = ttk.LabelFrame(right, text="Metaopis / poziomy abstrakcji")
        astlab_install_topology_controls(self, meta)
        meta.pack(fill=tk.X, pady=6)

        ttk.Label(meta, text="Tryb").grid(row=0, column=0, sticky="w")
        self.var_mode = tk.StringVar(value="depth-bucket")
        ttk.OptionMenu(meta, self.var_mode, "depth-bucket", "depth-bucket", "type", "defuse",
                       command=lambda _=None: self.render()).grid(row=0, column=1, sticky="w")

        ttk.Label(meta, text="bucket").grid(row=0, column=2, sticky="e")
        self.spin_bucket = tk.Spinbox(meta, from_=1, to=10, width=6, command=lambda: self.render())
        self.spin_bucket.delete(0, "end");
        self.spin_bucket.insert(0, "2")
        self.spin_bucket.grid(row=0, column=3, sticky="w")

        ttk.Label(meta, text="λ (abstrakcja)").grid(row=1, column=0, sticky="w")
        self.scale_lambda = tk.Scale(meta, from_=0.0, to=1.0, resolution=0.05,
                                     orient=tk.HORIZONTAL, command=lambda _v: self.render(), length=220)
        self.scale_lambda.set(0.0)
        self.scale_lambda.grid(row=1, column=1, columnspan=3, sticky="ew")

        self.var_show_anchors = tk.BooleanVar(value=True)
        self.var_show_hulls = tk.BooleanVar(value=False)
        ttk.Checkbutton(meta, text="Centroidy", variable=self.var_show_anchors, command=lambda: self.render()).grid(
            row=2, column=0, sticky="w")
        ttk.Checkbutton(meta, text="Obrysy bbox", variable=self.var_show_hulls, command=lambda: self.render()).grid(
            row=2, column=1, sticky="w")

        ttk.Button(meta, text="Okno poziomów…", command=self._open_meta_window).grid(row=2, column=3, sticky="e")
        self._meta_win = None

        # przyciski sceny
        btns = ttk.Frame(left);
        btns.pack(fill=tk.X, pady=6)
        ttk.Button(btns, text="Renderuj", command=lambda: self.render()).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="Reset widoku", command=self.scene.reset_view).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="Zapisz PNG", command=self._save_png).pack(side=tk.LEFT, padx=2)

        self.after(0, self.render)

    # pad
    def _toggle_S(self, lab):
        if lab in self.sel_S:
            self.sel_S.remove(lab)
        else:
            self.sel_S.add(lab)
        self.render()

    def _toggle_H(self, lab):
        if lab in self.sel_H:
            self.sel_H.remove(lab)
        else:
            self.sel_H.add(lab)
        self.render()

    def _clear_dirs(self):
        self.sel_S.clear();
        self.sel_H.clear();
        self.render()

    # status / auto-render
    def _set_status(self, msg, *, error=False):
        # aktualizacja paska statusu
        self.status.set(msg)
        # bezpieczna próba ustawienia tytułu okna najwyższego poziomu (bez managera matplotlib)
        try:
            top = self.winfo_toplevel()
            title = ("❌ " if error else "") + "AST → Kompas 3D"
            if hasattr(top, "title"):
                top.title(title)
        except Exception:
            # w środowiskach osadzonych może nie być toplevela — ignorujemy
            pass

    def _on_key(self, _e=None):
        if not self.var_auto_render.get(): return
        if self._render_job is not None: self.after_cancel(self._render_job)
        self._render_job = self.after(600, self.render)

    # I/O
    def _save_png(self):
        file = filedialog.asksaveasfilename(title="Zapisz obraz", defaultextension=".png",
                                            filetypes=[("PNG", "*.png")], initialfile="honey_ast_scene.png")
        if file: self.scene.fig.savefig(file, dpi=150)

    def _open_meta_window(self):
        if self._meta_win and tk.Toplevel.winfo_exists(self._meta_win):
            self._meta_win.lift();
            return
        self._meta_win = tk.Toplevel(self)
        self._meta_win.title("Poziomy abstrakcji — metaopis")
        self._meta_tree = ttk.Treeview(self._meta_win, columns=("size", "center", "bbox"), show="headings", height=18)
        for col, txt, w in (("size", "|C|", 60), ("center", "center (x,y,z)", 220), ("bbox", "bbox (xmin..zmax)", 320)):
            self._meta_tree.heading(col, text=txt);
            self._meta_tree.column(col, width=w, anchor="w")
        self._meta_tree.pack(fill=tk.BOTH, expand=True)
        ttk.Button(self._meta_win, text="Odśwież", command=self._refresh_meta_window).pack(anchor="e", pady=4, padx=4)

    def _refresh_meta_window(self):
        if not (self._meta_win and tk.Toplevel.winfo_exists(self._meta_win)):
            return
        tree = self._meta_tree
        for it in tree.get_children():
            tree.delete(it)
        if not hasattr(self, "_last_groups") or not self._last_groups:
            return
        for g in sorted(self._last_groups, key=lambda G: (-len(G["nodes"]), G["name"])):
            cx, cy, cz = g["center"];
            b = g["bbox"]
            tree.insert("", "end", values=(len(g["nodes"]),
                                           f"({cx:.2f},{cy:.2f},{cz:.2f})",
                                           f"({b[0]:.2f}..{b[1]:.2f}, {b[2]:.2f}..{b[3]:.2f}, {b[4]:.2f}..{b[5]:.2f})"))

    # render
    def render(self):
        # 1) Parsowanie kodu i stan UI
        code = self.txt.get("1.0", "end-1c")
        try:
            tree = ast.parse(code)
        except Exception as e:
            self.scene.clear("AST → Kompas 3D")
            self.scene.draw_compass()
            self._set_status(f"Błąd składni: {e}", error=True)
            self.scene.canvas.draw_idle()
            return

        self._set_status("OK — wyrenderowano", error=False)
        lam = float(self.scale_lambda.get())
        self._set_status(f"OK — λ={lam:.2f}, mode={self.var_mode.get()}", error=False)

        # 2) Budowa grafu i rzutowanie do sceny
        G = AstGraph().build_from_ast(tree)
        coords, labels, per_level = ast_to_scene_coords(tree)

        mode = self.var_mode.get()
        bucket = int(self.spin_bucket.get())
        groups, anchors, node2group = compute_meta_layers(
            tree, G, coords, per_level, mode=mode, bucket=bucket
        )
        self._last_groups = groups  # dla okna meta

        # Pozycje płynne: od detalu do centroidów (λ)
        coords_plot = {n: lerp(coords[n], anchors.get(n, coords[n]), lam) for n in coords}

        # 3) Parametry renderingu / ewolucji
        th_intra = float(getattr(self, "scale_thresh", tk.DoubleVar(value=0.6)).get())
        evolve = bool(getattr(self, "var_evolve", tk.BooleanVar(value=True)).get())

        show_S = self.var_show_S.get()
        show_H = self.var_show_H.get()
        show_labels = self.var_show_labels.get()

        lwS = float(self.scale_ls.get())
        lwH = float(self.scale_lh.get())
        sz = float(self.scale_pts.get())

        limit = int(self.spin_limit.get())
        zmax = int(self.spin_depth.get())

        id2node = {i: n for (n, i) in G.idmap.items()}

        # 4) Wybór widocznych węzłów zależnie od λ (kontrakcja do reprezentantów po progu)
        nodes_visible = select_nodes_for_lambda(
            per_level, coords_plot, zmax, limit, groups, lam,
            threshold=th_intra, evolve=evolve
        )

        # 5) Rysunek sceny
        self.scene.clear("AST → Kompas 3D")
        self.scene.draw_compass()

        # krawędzie i zbiór punktów do wyznaczenia granic
        xs, ys, zs = draw_edges_evolving(
            self.scene, G, id2node, node2group, coords_plot, nodes_visible,
            lam, th_intra, lwS, lwH, show_S, show_H, self.sel_S, self.sel_H, groups
        )

        # punkty (węzły)
        if nodes_visible:
            px = [coords_plot[n][0] for n in nodes_visible]
            py = [coords_plot[n][1] for n in nodes_visible]
            pz = [coords_plot[n][2] for n in nodes_visible]
            self.scene.ax.scatter(px, py, pz, s=sz)
            xs += px;
            ys += py;
            zs += pz

        # etykiety
        if show_labels:
            for n in nodes_visible:
                x, y, z = coords_plot[n]
                self.scene.ax.text(x, y, z + 0.3, labels[n], fontsize=7)

        # centroidy / obrysy klas (opcjonalnie)
        if self.var_show_anchors.get():
            for g in groups:
                cx, cy, cz = g["center"]
                self.scene.ax.scatter([cx], [cy], [cz], s=60, marker="^")
                self.scene.ax.text(cx, cy, cz + 0.4, g["name"], fontsize=7)

        if self.var_show_hulls.get():
            for g in groups:
                xmin, xmax, ymin, ymax, zmin, zmax_ = g["bbox"]
                X = [xmin, xmax];
                Y = [ymin, ymax];
                Z = [zmin, zmax_]
                edges = [
                    ((X[0], Y[0], Z[0]), (X[1], Y[0], Z[0])),
                    ((X[0], Y[1], Z[0]), (X[1], Y[1], Z[0])),
                    ((X[0], Y[0], Z[1]), (X[1], Y[0], Z[1])),
                    ((X[0], Y[1], Z[1]), (X[1], Y[1], Z[1])),
                    ((X[0], Y[0], Z[0]), (X[0], Y[1], Z[0])),
                    ((X[1], Y[0], Z[0]), (X[1], Y[1], Z[0])),
                    ((X[0], Y[0], Z[1]), (X[0], Y[1], Z[1])),
                    ((X[1], Y[0], Z[1]), (X[1], Y[1], Z[1])),
                    ((X[0], Y[0], Z[0]), (X[0], Y[0], Z[1])),
                    ((X[1], Y[0], Z[0]), (X[1], Y[0], Z[1])),
                    ((X[0], Y[1], Z[0]), (X[0], Y[1], Z[1])),
                    ((X[1], Y[1], Z[0]), (X[1], Y[1], Z[1])),
                ]
                for (p, q) in edges:
                    self.scene.ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], linewidth=0.6, alpha=0.6)

        # 6) Warstwa supergrafu między centroidami (dla wysokiego λ)
        if evolve and lam >= th_intra:
            super_nodes, super_edges = build_supergraph(groups, G, node2group)
            draw_supergraph_layer(self.scene.ax, groups, super_edges, lam)

        # 7) Limity i stabilizacja pudełka (bez pseudo-zoomu)
        xmax = max(xs + [6]);
        ymax = max(ys + [6]);
        zmaxp = max(zs + [10])
        self.scene.ax.set_xlim(-1, xmax + 1)
        self.scene.ax.set_ylim(-1, ymax + 1)
        self.scene.ax.set_zlim(-1, zmaxp + 1)
        set_box_aspect_equal(self.scene.ax)

        # 8) Final draw + ewentualne odświeżenie okna meta
        self.scene.canvas.draw_idle()
        try:
            self._refresh_meta_window()
        except Exception:
            pass


# ============================================================
# 7) ROOT i MAIN
# ============================================================

class RootApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Mosaic Lab — PRO (konsolidacja)")
        self.geometry("1360x900")
        self.nb = ttk.Notebook(self);
        self.nb.pack(fill=tk.BOTH, expand=True)
        self.nb.add(MosaicLabTab(self.nb), text="1) Mosaic Lab+")
        self.nb.add(MetaWorldTab(self.nb), text="2) Meta-świat (TO)")
        self.nb.add(Slice2DTab(self.nb), text="3) Przekroje 2D")
        self.nb.add(SimTab(self.nb), text="4) Symulacje (S/H)")
        self.nb.add(AstLabTab(self.nb), text="5) AST Lab")


if __name__ == "__main__":
    app = RootApp()
    app.mainloop()
