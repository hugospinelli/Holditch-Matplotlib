import bisect
import colorsys
import logging
import sys
import time
import tomllib

from dataclasses import dataclass
from typing import Self

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np

import Regions
from Ring import Ring

PRESETS_FILE = 'presets.toml'
CONFIG_FILE = 'config.toml'
LOG_FILE = 'holditch.log'

with open(CONFIG_FILE, 'rb') as f:
    config = tomllib.load(f)

PRESET = config['main'].get('preset', None)
if PRESET is not None:
    with open(PRESETS_FILE, 'rb') as f:
        presets_config = tomllib.load(f)
    for key, value in presets_config.get(PRESET, []).items():
        config[key].update(value)

A = config['main']['a']
B = config['main']['b']
CONTROL_POINTS = config['main']['control_points']

WIDTH = config['window']['width']
HEIGHT = config['window']['height']
HIDE_TOOLBAR = config['window']['hide_toolbar']

FPS = config['animation']['fps']
SPEED = config['animation']['speed']
STEPS_PER_FRAME = config['animation']['steps_per_frame']

BACKGROUND_COLOR = config['display']['background_color']
DATA_SCALE = config['display']['data_scale']
DISPLAY_SCALE = config['display']['display_scale']

SHORTCUT_DELETE_SELECTED_POINT = config['shortcuts']['delete_selected_point']
SHORTCUT_PAUSE = config['shortcuts']['pause']
SHORTCUT_HIDE_CONTROL_POINTS = config['shortcuts']['hide_control_points']
SHORTCUT_SAVE_INFO_TO_LOG_FILE = config['shortcuts']['save_info_to_log_file']

BEZIER_CIRCLE_RADIUS = config['bezier']['circle_radius']
BEZIER_CIRCLE_LINEWIDTH = config['bezier']['circle_linewidth']
BEZIER_CIRCLE_EDGECOLOR = config['bezier']['circle_edgecolor']
BEZIER_CIRCLE_FACECOLOR = config['bezier']['circle_facecolor']
BEZIER_SELECTED_FACECOLOR = config['bezier']['selected_facecolor']
BEZIER_OUTLINE_WIDTH = config['bezier']['outline_width']
BEZIER_OUTLINE_COLOR = config['bezier']['outline_color']

MAX_SEGMENT_LENGTH = config['curve']['max_segment_legth']
CURVE_LINE_WIDTH = config['curve']['line_width']
CURVE_LINE_COLOR = config['curve']['line_color']

CHORD_EDGE_RADIUS = config['chord']['edge_circles']['radius']
CHORD_EDGE_LINEWIDTH = config['chord']['edge_circles']['linewidth']
CHORD_EDGE_EDGECOLOR = config['chord']['edge_circles']['edgecolor']
CHORD_EDGE_FACECOLOR = config['chord']['edge_circles']['facecolor']
CHORD_CENTER_RADIUS = config['chord']['center_circle']['radius']
CHORD_CENTER_LINEWIDTH = config['chord']['center_circle']['linewidth']
CHORD_CENTER_EDGECOLOR = config['chord']['center_circle']['edgecolor']
CHORD_CENTER_FACECOLOR = config['chord']['center_circle']['facecolor']
CHORD_LINE_WIDTH = config['chord']['line']['width']
CHORD_LINE_COLOR = config['chord']['line']['color']

LOCUS_LINE_COLOR = config['locus']['line_color']
LOCUS_LINE_WIDTH = config['locus']['line_width']


# Save info to a log file when pressing 'l' (lower-case 'L')
logging.basicConfig(filename=LOG_FILE,
                    format='%(asctime)s - %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S', level=logging.INFO)


norm = np.linalg.norm


class Clock:
    def __init__(self, fps):
        self.fps = fps
        self.real_fps = fps
        self.dt = 1/fps
        self.t = time.time()
        
    def tick(self):
        elapsed_time = time.time() - self.t
        if elapsed_time < self.dt:
            plt.pause(self.dt - elapsed_time)
        self.real_fps = min(self.fps, 1/elapsed_time)
        self.t = time.time()


def get_winding_color(winding, max_winding=4):
    x = min(1, abs(winding)/max_winding)
    h = 0.7 if winding > 0 else 0
    return colorsys.hsv_to_rgb(h, 1, x)


class Bezier:
    def __init__(self, ax: plt.Axes):
        self.ax = ax
        self.circle_ps = Ring()  # coordinates of each circle
        self.selected_k: int | None = None  # index of selected circle
        self.dragged_k: int | None = None  # index of dragged circle
        self.dragged_dp = None  # select position relative to center
        self.curve_ps = Ring()

        if CONTROL_POINTS:
            self.circle_ps = Ring(np.array(pos) for pos in CONTROL_POINTS)
            self._update()
        
    def create(self, pos):
        if self.selected_k is None:
            self.selected_k = len(self.circle_ps)
            self.circle_ps.append(np.array(pos))
        else:
            self.selected_k += 1
            if self.selected_k == len(self.circle_ps):
                self.circle_ps.append(np.array(pos))
            else:
                self.circle_ps.insert(self.selected_k, np.array(pos))
        self._update()
        
    def select(self, pos, radius):
        pc = DATA_SCALE*np.array(pos)
        for k, p in enumerate(reversed(self.circle_ps)):
            dp = (pc[0] - p[0], pc[1] - p[1])
            if np.hypot(*dp) > radius:
                continue
            self.selected_k = self.dragged_k = len(self.circle_ps) - 1 - k
            self.dragged_dp = dp
            return
        self.create(pc)

    def release(self) -> bool:
        if self.dragged_k is None:
            return False
        self.dragged_k = self.dragged_dp = None
        return True

    def update_dragged(self, pos) -> bool:
        if self.dragged_k is None:
            return False
        pc = DATA_SCALE*np.array(pos)
        self.circle_ps[self.dragged_k] = np.array((pc[0] - self.dragged_dp[0],
                                                   pc[1] - self.dragged_dp[1]))
        self._update()
        return True

    def delete_selected(self) -> bool:
        if self.selected_k is None:
            return False
        del self.circle_ps[self.selected_k]
        self.selected_k = self.dragged_k = None
        self._update()
        return True

    @staticmethod
    def get_bezier(p1, p2, p3):
        min_dist = DATA_SCALE*MAX_SEGMENT_LENGTH
        a = norm(p2 - p1)
        b = norm(p3 - p2)
        n = max(1, round((a + b)/min_dist))
        bezier = []
        for k in range(n):
            t = k/n
            pa = p1 + t*(p2 - p1)
            pb = p2 + t*(p3 - p2)
            bezier.append(pa + t*(pb - pa))
        return bezier

    def _update(self):
        self.curve_ps = Ring()
        for p1, p2, p3 in self.circle_ps.triples():
            c1 = (p1 + p2)/2
            c3 = (p2 + p3)/2
            self.curve_ps.extend(self.get_bezier(c1, p2, c3))


@dataclass
class CurvePoint:
    p: np.ndarray
    """Position as a 2D vector."""
    
    k: int
    """Index of the segment where `p` lies, i.e., `p` sits between points
    `ps[k]` and `ps[k+1]` of a curve defined by a list of points `ps`."""
    
    d: float | None = None
    """Distance of `p` along the curve from the starting point."""


class Curve:
    def __init__(self, ax: plt.Axes, ps: Ring[np.ndarray]):
        self.ax = ax
        self.ps = ps
        self.n = len(ps)
        ds = np.cumsum([norm(p2 - p1) for p1, p2 in ps.pairs()])
        self.length = 0 if len(ds) == 0 else ds[-1]
        self.ds = Ring([0])
        self.ds.extend(ds[:-1])

    def get_d(self, p: np.ndarray, k: int) -> float:
        """Distance of `p` along the curve from the starting point."""
        return self.ds[k] + norm(p - self.ps[k])

    def get_dist(self, d1: float, d2: float) -> float:
        """
        Signed distance between two points along the curve, given their
        respective distances from the starting point.
        """
        s = self.length
        if s == 0:
            return 0
        return (d2 - d1 + s/2) % s - s/2  # goes from -s/2 to +s/2

    def get_cp(self, d: float) -> CurvePoint:
        """Get the point at distance `d` along the curve from the start."""
        d %= self.length
        k = bisect.bisect(self.ds, d) - 1
        v = self.ps[k + 1] - self.ps[k]
        if (v_length := norm(v)) == 0:
            p = self.ps[k]
        else:
            p = self.ps[k] + (d - self.ds[k])*v/v_length
        return CurvePoint(p, k, d)

    def slide(self, cp: CurvePoint, dist: float) -> CurvePoint:
        """Get the point at distance `dist` along the curve from `cp`."""
        return self.get_cp(cp.d + dist)
    
    @staticmethod
    def get_segment_intersections(
        p0: np.ndarray, r: float, p1: np.ndarray, p2: np.ndarray
    ) -> list[np.ndarray]:
        """
        Get a list of points with distance `r` to point `p0` which intersect
        the segment from point `p1` to `p2`.
        """
        x1, y1 = p1 - p0
        x2, y2 = p2 - p0
        d1 = np.hypot(x1, y1)
        d2 = np.hypot(x2, y2)
        # Check if both inside or both outside
        if (d1 - r)*(d2 - r) > 0:
            return []  # segment does not cross the circle
        dx = x2 - x1
        dy = y2 - y1
        dr = np.hypot(dx, dy)
        D = x1*y2 - x2*y1
        sign_dy = -1 if dy < 0 else 1
        discriminant = (r*dr)**2 - D**2
        if discriminant < 0 or dr == 0:
            return []
        intersections = []
        signs = (1,) if discriminant == 0 else (1, -1)
        for sign in signs:
            x = (D*dy + sign*sign_dy*dx*np.sqrt(discriminant))/dr**2
            y = (-D*dx + sign*abs(dy)*np.sqrt(discriminant))/dr**2
            px = np.array((x, y)) + p0
            if np.inner(px-p1, p2-p1) >= 0 and np.inner(px-p2, p1-p2) >= 0:
                intersections.append(px)
        return intersections

    def get_curve_intersections(
        self, r: float, cp_fixed: CurvePoint, cp0: CurvePoint,
        cutoff: float = np.inf
    ) -> list[CurvePoint]:
        """
        Get a list of points with distance `r` to `cp_fixed` which intersect
        a point in the curve at a distance `d` along the curve from `cp0`,
        where `d <= cutoff`.
        """
        cps = []
        if self.n < 2:
            return cps

        # Increasing k
        for dk in range(self.n//2):
            k = (cp0.k + dk) % self.n
            p1 = self.ps[k]
            p2 = self.ps[k + 1]
            if dk > 0 and self.get_dist(cp0.d, self.ds[k]) > cutoff:
                break
            for p in self.get_segment_intersections(cp_fixed.p, r, p1, p2):
                d = self.get_d(p, k)
                if self.get_dist(cp0.d, d) <= cutoff:
                    cps.append(CurvePoint(p, k, d))
               
        # Decreasing k (commented with asterisk where different from before)
        for dk in range(self.n//2):
            k = (cp0.k - dk - 1) % self.n  # *
            p1 = self.ps[k]
            p2 = self.ps[k + 1]
            if self.get_dist(self.ds[k + 1], cp0.d) > cutoff:  # *
                break
            for p in self.get_segment_intersections(cp_fixed.p, r, p1, p2):
                d = self.get_d(p, k)
                if self.get_dist(d, cp0.d) <= cutoff:  # *
                    cps.append(CurvePoint(p, k, d))

        return cps
        

class Chord:
    def __init__(self, curve: Curve, cpa: CurvePoint, cpb: CurvePoint,
                 a: float, b: float):
        self.curve = curve
        self.cpa = cpa
        self.cpb = cpb
        self.a = a
        self.b = b
        self.ab = cpb.p - cpa.p
        self.pc = self.cpa.p + a*self.ab/(a + b)

    def is_ahead(self, other: Self) -> bool:
        """Check if `other` is ahead of `self`."""
        da12 = self.curve.get_dist(self.cpa.d, other.cpa.d)
        db12 = self.curve.get_dist(self.cpb.d, other.cpb.d)
        if da12 > 0 and db12 > 0:
            return True
        a12 = other.cpa.p - self.cpa.p
        b12 = other.cpb.p - self.cpb.p
        rot_12 = np.cross(da12*a12, db12*b12)
        rot_ab = np.cross(self.ab, other.ab)
        if rot_12 == rot_ab == 0:
            return da12 > 0
        return rot_12*rot_ab > 0  # same sign

    def slide(self, dist: float) -> Self | None:
        for sign in (1, -1):
            # Slide `cpa` by exactly `sign*dist`
            cpa2 = self.curve.slide(self.cpa, sign*dist)
            for cpb2 in self.curve.get_curve_intersections(
                self.a + self.b, cpa2, self.cpb, abs(dist)
            ):
                new_chord = Chord(self.curve, cpa2, cpb2, self.a, self.b)
                if self.is_ahead(new_chord):
                    if dist > 0:
                        return new_chord
                elif dist < 0:
                    return new_chord

            # Slide `cpb` by exactly `sign*dist`
            cpb2 = self.curve.slide(self.cpb, sign*dist)
            for cpa2 in self.curve.get_curve_intersections(
                self.a + self.b, cpb2, self.cpa, abs(dist)
            ):
                new_chord = Chord(self.curve, cpa2, cpb2, self.a, self.b)
                if self.is_ahead(new_chord):
                    if dist > 0:
                        return new_chord
                elif dist < 0:
                    return new_chord
        
        return None


class Locus:
    speed = SPEED*DATA_SCALE
    
    def __init__(self, ax: plt.Axes, curve: Curve,
                 a: float, b: float):
        self.ax = ax
        self.curve = curve
        self.a = a*DATA_SCALE
        self.b = b*DATA_SCALE
        self.r = self.a + self.b
        self.ps = Ring()
        self.closed = False
        self.chord = None
        self.start_chord = None
    
        if len(curve.ps) < 2:
            return
        ka = np.argmin([p[1] for p in curve.ps]) + 1
        cpa = CurvePoint(curve.ps[ka], ka, curve.ds[ka])
        cpbs = curve.get_curve_intersections(self.r, cpa, cpa)
        if len(cpbs) == 0:
            return
        cpb = cpbs[1]
        self.chord = Chord(curve, cpa, cpb, self.a, self.b)
        self.start_chord = self.chord

    def _get_line(self):
        return plt.Line2D(
            (0, 0), (0, 0),
            color=self.line_color,
            linewidth=self.line_width,
            solid_capstyle='round',
            solid_joinstyle='round',
            animated=True,
            antialiased=True,
            visible=False,
            #zorder=??,
        )

    def update(self, dt=1/FPS):
        if self.chord is None or dt == 0:
            return
        dist = dt*self.speed
        for _ in range(10):
            if (new_chord := self.chord.slide(dist)) is None:
                dist /= 2
        if new_chord is None:
            return
        if self.closed:
            self.chord = new_chord
            return
        dist_start = self.curve.get_dist(new_chord.cpa.d,
                                         self.start_chord.cpa.d)
        if (
            abs(dist_start) <= dist
            and self.chord.is_ahead(self.start_chord)
            and self.start_chord.is_ahead(new_chord)
        ):
            self.closed = True
            self.chord = new_chord
            return
        self.ps.append(new_chord.pc)
        self.chord = new_chord


###############################################################################
# Artists
###############################################################################


class BezierArtist:
    circle_radius = BEZIER_CIRCLE_RADIUS * DATA_SCALE
    circle_linewidth = BEZIER_CIRCLE_LINEWIDTH * DISPLAY_SCALE
    circle_edgecolor = BEZIER_CIRCLE_EDGECOLOR
    circle_facecolor = BEZIER_CIRCLE_FACECOLOR
    selected_facecolor = BEZIER_SELECTED_FACECOLOR
    
    outline_width = BEZIER_OUTLINE_WIDTH * DISPLAY_SCALE
    outline_color = BEZIER_OUTLINE_COLOR
    
    def __init__(self, ax: plt.Axes, bezier: Bezier):
        self.ax = ax
        self.bezier = Bezier
        self.outline = self._get_outline()
        self.circles = []
        ax.add_artist(self.outline)
        self.visible = True

    def _get_outline(self):
        return plt.Line2D(
            (0, 0), (0, 0),
            color=self.outline_color,
            linewidth=self.outline_width,
            solid_capstyle='round',
            solid_joinstyle='round',
            animated=True,
            antialiased=True,
            visible=False,
            zorder=40,
        )

    def _get_circle(self):
        return plt.Circle(
            (0, 0),
            self.circle_radius,
            linewidth=self.circle_linewidth,
            edgecolor=self.circle_edgecolor,
            facecolor=self.circle_facecolor,
            animated=True,
            antialiased=True,
            visible=True,
            zorder=50,
        )

    def toggle_visibility(self):
        if self.visible:
            self.bezier.release()
        self.visible = not self.visible
        if len(self.bezier.circle_ps) < 2:
            self.outline.set_visible(False)
        else:
            self.outline.set_visible(self.visible)
        for circle in self.circles:
            circle.set_visible(self.visible)
        
    def select(self, mouse_pos) -> bool:
        if not self.visible:
            return False
        self.bezier.select(mouse_pos, self.circle_radius)
        return True

    def release(self) -> bool:
        if not self.visible:
            return False
        return self.bezier.release()

    def update_dragged(self, mouse_pos) -> bool:
        if not self.visible:
            return False
        return self.bezier.update_dragged(mouse_pos)

    def delete_selected(self) -> bool:
        if not self.visible:
            return False
        return self.bezier.delete_selected()

    def update(self, bezier: Bezier | None):
        self.bezier = bezier
        if len(bezier.curve_ps) < 2:
            self.outline.set_visible(False)
        elif self.visible:
            self.outline.set_visible(True)
        if len(bezier.curve_ps) >= 2:
            xs = [p[0] for p in bezier.circle_ps]
            ys = [p[1] for p in bezier.circle_ps]
            xs.append(bezier.circle_ps[0][0])
            ys.append(bezier.circle_ps[0][1])
            self.outline.set_xdata(xs)
            self.outline.set_ydata(ys)
        
        n0, n = len(self.circles), len(bezier.circle_ps)
        for k in range(n0, n):
            new_circle = self._get_circle()
            self.circles.append(new_circle)
            self.ax.add_patch(new_circle)
        for k in reversed(range(n, n0)):
            self.circles[k].remove()
            del self.circles[k]
        for k, (circle, p) in enumerate(zip(self.circles, bezier.circle_ps)):
            circle.set_center(p)
            if k == bezier.selected_k:
                circle.set_facecolor(self.selected_facecolor)
            else:
                circle.set_facecolor(self.circle_facecolor)
        
    def draw(self):
        self.ax.draw_artist(self.outline)
        for circle in self.circles:
            self.ax.draw_artist(circle)

    def delete(self):
        self.outline.remove()
        for circle in self.circles:
            circle.remove()


class CurveArtist:
    line_width = CURVE_LINE_WIDTH * DISPLAY_SCALE
    line_color = CURVE_LINE_COLOR
    
    def __init__(self, ax: plt.Axes):
        self.ax = ax
        self.line = self._get_line()
        ax.add_artist(self.line)

    def _get_line(self):
        return plt.Line2D(
            (0, 0), (0, 0),
            color=self.line_color,
            linewidth=self.line_width,
            solid_capstyle='round',
            solid_joinstyle='round',
            animated=True,
            antialiased=True,
            visible=False,
            zorder=30,
        )

    def update(self, curve: Curve | None):
        is_visible = len(curve.ps) >= 2
        self.line.set_visible(is_visible)
        if not is_visible:
            return
        xs = [p[0] for p in curve.ps]
        ys = [p[1] for p in curve.ps]
        xs.append(curve.ps[0][0])
        ys.append(curve.ps[0][1])
        self.line.set_xdata(xs)
        self.line.set_ydata(ys)
        
    def draw(self):
        self.ax.draw_artist(self.line)
    
    def delete(self):
        self.line.remove()


class ChordArtist:
    edge_radius = CHORD_EDGE_RADIUS * DATA_SCALE
    edge_linewidth = CHORD_EDGE_LINEWIDTH * DISPLAY_SCALE
    edge_edgecolor = CHORD_EDGE_EDGECOLOR
    edge_facecolor = CHORD_EDGE_FACECOLOR
    
    center_radius = CHORD_CENTER_RADIUS * DATA_SCALE
    center_linewidth = CHORD_CENTER_LINEWIDTH * DISPLAY_SCALE
    center_edgecolor = CHORD_CENTER_EDGECOLOR
    center_facecolor = CHORD_CENTER_FACECOLOR
    
    line_width = CHORD_LINE_WIDTH * DISPLAY_SCALE
    line_color = CHORD_LINE_COLOR

    def __init__(self, ax: plt.Axes):
        self.ax = ax
        self.line = self._get_line()
        self.circle_a = self._get_edge_circle()
        self.circle_b = self._get_edge_circle()
        self.circle_c = self._get_center_circle()

        ax.add_artist(self.line)
        ax.add_patch(self.circle_a)
        ax.add_patch(self.circle_b)
        ax.add_patch(self.circle_c)

    def _get_line(self):
        return plt.Line2D(
            (0, 0), (0, 0),
            color=self.line_color,
            linewidth=self.line_width,
            solid_capstyle='round',
            solid_joinstyle='round',
            animated=True,
            antialiased=True,
            visible=False,
            zorder=60,
        )

    def _get_edge_circle(self):
        return plt.Circle(
            (0, 0),
            self.edge_radius,
            linewidth=self.edge_linewidth,
            edgecolor=self.edge_edgecolor,
            facecolor=self.edge_facecolor,
            animated=True,
            antialiased=True,
            visible=False,
            zorder=70,
        )

    def _get_center_circle(self):
        return plt.Circle(
            (0, 0),
            self.center_radius,
            linewidth=self.center_linewidth,
            edgecolor=self.center_edgecolor,
            facecolor=self.center_facecolor,
            animated=True,
            antialiased=True,
            visible=False,
            zorder=80,
        )

    def update(self, chord: Chord | None):
        is_visible = chord is not None
        self.line.set_visible(is_visible)
        self.circle_a.set_visible(is_visible)
        self.circle_b.set_visible(is_visible)
        self.circle_c.set_visible(is_visible)
        
        if is_visible:
            self.line.set_xdata((chord.cpa.p[0], chord.cpb.p[0]))
            self.line.set_ydata((chord.cpa.p[1], chord.cpb.p[1]))
            self.circle_a.set_center(chord.cpa.p)
            self.circle_b.set_center(chord.cpb.p)
            self.circle_c.set_center(chord.pc)
        
    def draw(self):
        self.ax.draw_artist(self.line)
        self.ax.draw_artist(self.circle_a)
        self.ax.draw_artist(self.circle_b)
        self.ax.draw_artist(self.circle_c)

    def delete(self):
        self.line.remove()
        self.circle_a.remove()
        self.circle_b.remove()
        self.circle_c.remove()


class LocusArtist:
    line_color = LOCUS_LINE_COLOR
    line_width = LOCUS_LINE_WIDTH * DISPLAY_SCALE
    
    def __init__(self, ax: plt.Axes):
        self.ax: plt.Axes = ax
        self.closed: bool = False
        self.line: list[plt.Line2D] = self._get_line()
        self.areas: list[plt.Polygon] = []
        self.total_area: float | None = None
        ax.add_artist(self.line)

    def _get_line(self):
        return plt.Line2D(
            (0, 0), (0, 0),
            color=self.line_color,
            linewidth=self.line_width,
            solid_capstyle='round',
            solid_joinstyle='round',
            animated=True,
            antialiased=True,
            visible=False,
            zorder=20,
        )

    def _get_polygon(self, xy, color):
        return plt.Polygon(
            xy,
            facecolor=color,
            edgecolor=color,
            linewidth=0.5,
            fill=True,
            animated=True,
            antialiased=True,
            visible=True,
            zorder=10,
        )

    def update(self, locus: Locus | None):
        is_visible = len(locus.ps) > 1
        self.line.set_visible(is_visible)
        if not is_visible:
            return
        
        xs = [p[0] for p in locus.ps]
        ys = [p[1] for p in locus.ps]
        
        # Close path
        if locus.closed:
            if self.closed:
                return
            xs.append(locus.ps[0][0])
            ys.append(locus.ps[0][1])
            paths, windings, self.total_area = Regions.split_paths(
                locus.curve.ps, locus.ps)
            for path, winding in zip(paths, windings):
                color = get_winding_color(winding)
                area = self._get_polygon(path, color)
                self.areas.append(area)
                self.ax.add_patch(area)
            self.closed = True
        elif self.closed:
            self.delete_areas()
        
        self.line.set_xdata(xs)
        self.line.set_ydata(ys)
        
    def draw(self):
        for area in self.areas:
            self.ax.draw_artist(area)
        self.ax.draw_artist(self.line)

    def delete_areas(self):
        for area in self.areas:
            area.remove()
        self.areas = []
        self.closed = False
        self.total_area = None

    def delete(self):
        self.delete_areas()
        self.line.remove()


###############################################################################
# Animation
###############################################################################


class Animation:
    a, b = A, B
    
    def __init__(self, ax, canvas):
        self.ax = ax
        self.canvas = canvas
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)

        self.bezier = Bezier(ax)
        self.curve = Curve(ax, self.bezier.curve_ps)
        self.locus = Locus(ax, self.curve, self.a, self.b)

        self.bezier_artist = BezierArtist(ax, self.bezier)
        self.curve_artist = CurveArtist(ax)
        self.locus_artist = LocusArtist(ax)
        self.chord_artist = ChordArtist(ax)
        
        self.restart_slider = False
        self.paused = False
        self.recording = False
        self._frame_count = 0
        
        canvas.mpl_connect('draw_event', self.on_draw)
        canvas.mpl_connect('button_press_event', self.on_button_press)
        canvas.mpl_connect('button_release_event', self.on_button_release)
        canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        canvas.mpl_connect('key_press_event', self.on_key_press)

    def on_draw(self, event):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)

    def on_button_press(self, event):
        mouse_pos = (event.xdata, event.ydata)
        if None in mouse_pos:
            return
        if self.bezier_artist.select(mouse_pos):
            self.restart_slider = True

    def on_button_release(self, event):
        if self.bezier_artist.release():
            pass

    def on_mouse_move(self, event):
        """Callback for mouse movements."""
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        mouse_pos = (event.xdata, event.ydata)
        if None in mouse_pos:
            return
        if self.bezier_artist.update_dragged(mouse_pos):
            self.restart_slider = True

    def on_key_press(self, event):
        if event.key == SHORTCUT_DELETE_SELECTED_POINT:
            if self.bezier_artist.delete_selected():
                self.restart_slider = True
        elif event.key == SHORTCUT_PAUSE:
            self.paused = not self.paused
        elif event.key == SHORTCUT_HIDE_CONTROL_POINTS:
            self.bezier_artist.toggle_visibility()
        # Coordinates of the control points
        elif event.key == SHORTCUT_SAVE_INFO_TO_LOG_FILE:
            self.log()
        elif event.key == 'r':
            self.recording = not self.recording
            if self.recording:
                self._frame_count = 0
                self.restart_slider = True
                self.paused = False

    def log(self):
        control_points_str = str(list(list(p) for p in self.bezier.circle_ps))
        logging.info(f'Control points = {control_points_str}')
        logging.info(f'a, b = {self.a}, {self.b}')
        if self.locus_artist.total_area is None:
            logging.info('Total area not yet determined.')
        else:
            logging.info(f'Total area = {self.locus_artist.total_area}')

    def update(self, dt=1/FPS):
        if self.restart_slider:
            self.curve = Curve(self.ax, self.bezier.curve_ps)
            self.locus = Locus(self.ax, self.curve, self.a, self.b)
            self.locus_artist.delete_areas()
        else:
            if self.paused:
                return
            self.locus.update(dt)
        self.restart_slider = False

    def draw(self):
        if self.recording:
            if self.locus_artist.closed:
                self._frame_count -= 1
            else:
                self._frame_count += 1
            if self._frame_count == 0:
                self.recording = False
        
        self.bezier_artist.update(self.bezier)
        self.curve_artist.update(self.curve)
        self.locus_artist.update(self.locus)
        self.chord_artist.update(self.locus.chord)

        self.canvas.restore_region(self.background)

        self.locus_artist.draw()
        self.curve_artist.draw()
        self.bezier_artist.draw()
        self.chord_artist.draw()
        
        self.canvas.blit(self.ax.bbox)
        self.canvas.flush_events()


def main():
    disabled_keymaps = ['xscale', 'yscale', 'zoom', 'fullscreen', 'pan']
    for keymap in disabled_keymaps:
        plt.rcParams[f'keymap.{keymap}'] = []
    plt.rcParams['keymap.home'] = ['home']

    if HIDE_TOOLBAR:
        plt.rcParams['toolbar'] = 'None'

    #FFMpegWriter = matplotlib.animation.FFMpegFileWriter
    FFMpegWriter = matplotlib.animation.writers['ffmpeg']
    metadata = dict(title="Holditch's Theorem", artist='Matplotlib')
    writer = matplotlib.animation.FFMpegFileWriter(
        fps=FPS, metadata=metadata,
        codec='libx264', extra_args=['-pix_fmt', 'yuv420p']
    )

    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    fig, ax = plt.subplots(figsize=(WIDTH*px, HEIGHT*px))
    ax.axis('square')
    aspect_ratio = WIDTH/HEIGHT
    ax.set_xlim((-100*aspect_ratio, 100*aspect_ratio))
    ax.set_ylim((-100, 100))
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.set_facecolor(BACKGROUND_COLOR)
    plt.get_current_fig_manager().set_window_title("Holditch's Theorem")
    
    anim = Animation(ax, fig.canvas)
    
    plt.show(block=False)
    plt.pause(0.1)

    clock = Clock(FPS)
    while plt.get_fignums():
        for _ in range(STEPS_PER_FRAME):
            anim.update(dt=(1/FPS)/STEPS_PER_FRAME)
        anim.draw()
        clock.tick()
        
        if anim.recording:
            video_file_name = f"Holditch_{time.strftime('%Y%m%dT%H%M%S')}.mp4"
            with writer.saving(fig, video_file_name, 200):
                while plt.get_fignums() and anim.recording:
                    for _ in range(STEPS_PER_FRAME):
                        anim.update(dt=(1/FPS)/STEPS_PER_FRAME)
                    anim.draw()
                    writer.grab_frame()
                    clock.tick()


if __name__ == '__main__':
    main()
