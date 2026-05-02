from __future__ import annotations

"""
TileCut Engineering System v3

НОВОЕ в v3:
- Многокомнатность: до 8 комнат в одном проекте, общая закупка с учётом остатков
- Нестандартные формы: произвольный полигон (Г, Т, скосы) через список точек X Y
- Печатная карта реза: PDF landscape A4/A3, масштаб, легенда, номера, размеры
- Калькулятор закупки: плитка, упаковки, клей, затирка, гидроизоляция — итоговая смета

Запуск:
    pip install streamlit pandas numpy plotly openpyxl reportlab ezdxf shapely
    streamlit run tilecut_streamlit.py
"""

from dataclasses import dataclass, field
from enum import Enum
from io import BytesIO
from math import ceil
from typing import Iterable
import itertools
import os
import tempfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ─── Shapely ────────────────────────────────────────────────────────────────
SHAPELY_AVAILABLE = True
SHAPELY_ERROR     = None

try:
    from shapely import affinity
    from shapely.geometry import Polygon, box, MultiPolygon
    from shapely.ops import unary_union
except Exception as exc:
    SHAPELY_AVAILABLE = False
    SHAPELY_ERROR     = exc
    Polygon      = object
    MultiPolygon = object
    box = affinity = unary_union = None

# ─── CONFIG ──────────────────────────────────────────────────────────────────
APP_TITLE            = "TileCut Engineering System v3"
SMALL_CUT_DEFAULT_MM = 100.0
EPS                  = 0.001
WASTE_FACTOR         = 0.10

COLOR_BG     = "#0A0C12"
COLOR_PANEL  = "#10131E"
COLOR_GRID   = "#313A55"
COLOR_TEXT   = "#F3F6FF"
COLOR_DIM    = "#8C96B3"
COLOR_FULL   = "rgba(190,190,180,0.32)"
COLOR_CUT    = "rgba(232,160,32,0.48)"
COLOR_RISK   = "rgba(231,76,60,0.58)"
COLOR_REUSED = "rgba(46,204,113,0.45)"
COLOR_ROOM   = "rgba(20,23,42,0.65)"
COLOR_DOOR   = "rgba(80,55,28,0.75)"
COLOR_ACCENT = "#E8A020"

ADHESIVE_KG_PER_M2 = 4.5
GROUT_KG_PER_M2    = 0.3
UNDERLAY_M2_PER_M2 = 1.0


# ============================================================
# MODELS
# ============================================================

class PatternType(str, Enum):
    GRID     = "grid"
    OFFSET   = "offset"
    DIAGONAL = "diagonal"
    MODULAR  = "modular"


class OptimizeGoal(str, Enum):
    NO_SMALL_CUTS = "no_small_cuts"
    MIN_WASTE     = "minimum_waste"
    SYMMETRY      = "symmetry"
    CUT_REUSE     = "cut_reuse"


class RoomShapeType(str, Enum):
    RECTANGLE = "rectangle"
    CUSTOM    = "custom"


@dataclass(frozen=True)
class TileFormat:
    width: float
    height: float
    name: str = "Tile"

    @property
    def area_m2(self) -> float:
        return self.width * self.height / 1_000_000


@dataclass
class RoomSpec:
    name: str = "Room"
    width: float = 3000.0
    height: float = 2000.0
    shape_type: RoomShapeType = RoomShapeType.RECTANGLE
    custom_points: list = field(default_factory=list)

    def polygon(self) -> Polygon:
        if self.shape_type == RoomShapeType.CUSTOM and len(self.custom_points) >= 3:
            return Polygon(self.custom_points)
        return box(0, 0, self.width, self.height)

    @property
    def area_m2(self) -> float:
        return self.polygon().area / 1_000_000

    @property
    def bbox_width(self) -> float:
        minx, _, maxx, _ = self.polygon().bounds
        return maxx - minx

    @property
    def bbox_height(self) -> float:
        _, miny, _, maxy = self.polygon().bounds
        return maxy - miny


@dataclass(frozen=True)
class Obstacle:
    name: str
    x: float
    y: float
    width: float
    height: float

    def polygon(self) -> Polygon:
        return box(self.x, self.y, self.x + self.width, self.y + self.height)


@dataclass
class LayoutParams:
    pattern: PatternType
    offset_ratio: float = 0.5
    angle_deg: float = 0.0
    joint_mm: float = 2.0
    start_x: float = 0.0
    start_y: float = 0.0
    small_cut_mm: float = SMALL_CUT_DEFAULT_MM
    modular_sequence: list = field(default_factory=list)
    optimize_goal: OptimizeGoal = OptimizeGoal.NO_SMALL_CUTS


@dataclass
class TilePiece:
    index: int
    source_tile_id: int
    polygon: Polygon
    bbox_w: float
    bbox_h: float
    area_m2: float
    is_full: bool
    is_cut: bool
    is_small_cut: bool
    is_obstacle_cut: bool
    is_rectangular: bool
    pattern: str
    x: float
    y: float
    reused_from_offcut: bool = False

    @property
    def size_label(self) -> str:
        return f"{self.bbox_w:.0f}x{self.bbox_h:.0f}"

    @property
    def cut_type(self) -> str:
        if self.is_full:            return "целая"
        if self.reused_from_offcut: return "reuse остаток"
        if self.is_obstacle_cut:    return "подрезка по проёму"
        return "подрезка"


@dataclass
class LayoutScore:
    waste_m2: float
    waste_pct: float
    full_tiles: int
    cut_pieces: int
    small_cuts: int
    symmetry_error: float
    reusable_pairs: int
    score: float
    tiles_to_purchase: int


@dataclass
class LayoutResult:
    room: RoomSpec
    tile: TileFormat
    params: LayoutParams
    pieces: list
    score: LayoutScore
    obstacles: list = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "№": p.index, "Тип": p.cut_type, "Размер": p.size_label,
            "Ширина": round(p.bbox_w,1), "Высота": round(p.bbox_h,1),
            "X": round(p.x,1), "Y": round(p.y,1),
            "Площадь м²": round(p.area_m2,4),
            "Прямоуг.": "ДА" if p.is_rectangular else "НЕТ",
            "Мелкая": "ДА" if p.is_small_cut else "",
            "Reuse": "ДА" if p.reused_from_offcut else "",
        } for p in self.pieces])


# ============================================================
# PURCHASE CALCULATOR
# ============================================================

@dataclass
class PurchaseCalc:
    tile_price_per_m2: float   = 0.0
    tiles_per_pack: float      = 1.0
    pack_area_m2: float        = 1.0
    adhesive_price_per_kg: float  = 0.0
    grout_price_per_kg: float     = 0.0
    underlay_price_per_m2: float  = 0.0
    currency: str              = "грн"

    def calculate(self, results: list) -> dict:
        total_area   = sum(r.room.area_m2 for r in results)
        total_tiles  = sum(r.score.tiles_to_purchase for r in results)
        cross_reuse  = sum(r.score.reusable_pairs for r in results)
        net_tiles    = max(total_tiles - cross_reuse, total_tiles)
        packs        = ceil(net_tiles / self.tiles_per_pack) if self.tiles_per_pack > 0 else 0
        bought_m2    = packs * self.pack_area_m2

        tile_cost    = bought_m2    * self.tile_price_per_m2
        adh_cost     = total_area   * ADHESIVE_KG_PER_M2 * self.adhesive_price_per_kg
        grout_cost   = total_area   * GROUT_KG_PER_M2    * self.grout_price_per_kg
        under_cost   = total_area   * UNDERLAY_M2_PER_M2 * self.underlay_price_per_m2
        total_cost   = tile_cost + adh_cost + grout_cost + under_cost
        cur          = self.currency

        return {
            "Комнат":                         len(results),
            "Площадь укладки, м²":            round(total_area, 2),
            "Плиток нетто":                   total_tiles,
            "Cross-reuse экономия, шт":       cross_reuse,
            "Плиток к закупке":               net_tiles,
            "Упаковок":                       packs,
            "Закуплено, м²":                  round(bought_m2, 2),
            f"Плитка, {cur}":                 round(tile_cost, 2),
            f"Клей ({ADHESIVE_KG_PER_M2} кг/м²), {cur}": round(adh_cost, 2),
            f"Затирка ({GROUT_KG_PER_M2} кг/м²), {cur}": round(grout_cost, 2),
            f"Гидроизоляция, {cur}":          round(under_cost, 2),
            f"ИТОГО, {cur}":                  round(total_cost, 2),
        }


# ============================================================
# GEOMETRY ENGINE
# ============================================================

class GeometryEngine:

    @staticmethod
    def work_area(room: RoomSpec, obstacles: list) -> Polygon:
        area = room.polygon()
        for obs in obstacles:
            area = area.difference(obs.polygon())
        return area

    @staticmethod
    def tile_polygon(x, y, w, h, angle_deg=0.0) -> Polygon:
        p = box(x, y, x+w, y+h)
        if abs(angle_deg) > EPS:
            p = affinity.rotate(p, angle_deg, origin=(x+w/2, y+h/2), use_radians=False)
        return p

    @staticmethod
    def clip_to_area(tile: Polygon, area: Polygon):
        c = tile.intersection(area)
        return None if (c.is_empty or c.area < 1) else c

    @staticmethod
    def bbox_size(poly: Polygon):
        minx, miny, maxx, maxy = poly.bounds
        return minx, miny, maxx-minx, maxy-miny

    @staticmethod
    def is_full_piece(piece: Polygon, tw, th) -> bool:
        exp = tw * th
        return abs(piece.area - exp) / exp < 0.005

    @staticmethod
    def is_rectangular(poly: Polygon) -> bool:
        minx, miny, maxx, maxy = poly.bounds
        bb = (maxx-minx) * (maxy-miny)
        return bb > EPS and abs(poly.area - bb) / bb < 0.01


# ============================================================
# PATTERNS
# ============================================================

class GridPattern:
    def generate_tiles(self, room: RoomSpec, tile: TileFormat, params: LayoutParams, margin=2):
        sx, sy = params.start_x, params.start_y
        bw, bh = room.bbox_width, room.bbox_height
        stepx  = tile.width  + params.joint_mm
        stepy  = tile.height + params.joint_mm
        idx = 1
        y = sy - tile.height * margin
        while y <= bh + tile.height * margin:
            x = sx - tile.width * margin
            while x <= bw + tile.width * margin:
                yield idx, GeometryEngine.tile_polygon(x, y, tile.width, tile.height, params.angle_deg)
                idx += 1; x += stepx
            y += stepy


class OffsetPattern:
    def generate_tiles(self, room: RoomSpec, tile: TileFormat, params: LayoutParams, margin=2):
        bw, bh = room.bbox_width, room.bbox_height
        stepx  = tile.width  + params.joint_mm
        stepy  = tile.height + params.joint_mm
        offset = tile.width  * params.offset_ratio
        idx = 1; row = 0
        y = params.start_y - tile.height * margin
        while y <= bh + tile.height * margin:
            shift = offset if row % 2 else 0.0
            x = params.start_x - tile.width * margin
            while x <= bw + tile.width * margin:
                yield idx, GeometryEngine.tile_polygon(x+shift, y, tile.width, tile.height, params.angle_deg)
                idx += 1; x += stepx
            y += stepy; row += 1


class DiagonalPattern:
    def generate_tiles(self, room: RoomSpec, tile: TileFormat, params: LayoutParams, margin=4):
        p45 = LayoutParams(pattern=params.pattern, offset_ratio=params.offset_ratio,
                           angle_deg=45.0, joint_mm=params.joint_mm,
                           start_x=params.start_x, start_y=params.start_y,
                           small_cut_mm=params.small_cut_mm, optimize_goal=params.optimize_goal)
        yield from GridPattern().generate_tiles(room, tile, p45, margin=margin)


class ModularPattern:
    def generate_tiles(self, room: RoomSpec, tile: TileFormat, params: LayoutParams, margin=2):
        seq = params.modular_sequence or [
            tile,
            TileFormat(tile.width/2, tile.height, "Half"),
            TileFormat(tile.width, tile.height/2, "Half-Y"),
        ]
        maxw = max(t.width  for t in seq)
        maxh = max(t.height for t in seq)
        bw, bh = room.bbox_width, room.bbox_height
        idx = 1; row = 0
        y = params.start_y - maxh * margin
        while y <= bh + maxh * margin:
            col = 0; x = params.start_x - maxw * margin
            while x <= bw + maxw * margin:
                fmt = seq[(row+col) % len(seq)]
                yield idx, GeometryEngine.tile_polygon(x, y, fmt.width, fmt.height, params.angle_deg)
                idx += 1; x += fmt.width + params.joint_mm; col += 1
            y += maxh + params.joint_mm; row += 1


PATTERN_FACTORY = {
    PatternType.GRID:     GridPattern,
    PatternType.OFFSET:   OffsetPattern,
    PatternType.DIAGONAL: DiagonalPattern,
    PatternType.MODULAR:  ModularPattern,
}


# ============================================================
# CUT MAP ENGINE
# ============================================================

class CutMapEngine:

    @staticmethod
    def estimate_offcuts(result: LayoutResult) -> pd.DataFrame:
        rows = []
        tw, th = result.tile.width, result.tile.height
        for p in result.pieces:
            if not p.is_cut or not p.is_rectangular: continue
            rw = tw - p.bbox_w; rh = th - p.bbox_h
            if rw >= result.params.small_cut_mm:
                rows.append({"От №": p.index, "Остаток": f"{rw:.0f}x{th:.0f}",
                             "Ш": round(rw,1), "В": round(th,1),
                             "м²": round(rw*th/1e6, 4)})
            if rh >= result.params.small_cut_mm:
                rows.append({"От №": p.index, "Остаток": f"{tw:.0f}x{rh:.0f}",
                             "Ш": round(tw,1), "В": round(rh,1),
                             "м²": round(tw*rh/1e6, 4)})
        return pd.DataFrame(rows)

    @staticmethod
    def mark_reuse_candidates(pieces: list, tile: TileFormat, small_cut_mm: float) -> int:
        rect_cuts = [p for p in pieces if p.is_cut and not p.is_small_cut and p.is_rectangular]
        offcuts = []
        for p in rect_cuts:
            rw = tile.width  - p.bbox_w
            rh = tile.height - p.bbox_h
            if rw >= small_cut_mm: offcuts.append((rw, tile.height))
            if rh >= small_cut_mm: offcuts.append((tile.width, rh))
        used = set(); reused = 0
        for p in sorted(rect_cuts, key=lambda x: x.area_m2):
            for i, (ow, oh) in enumerate(offcuts):
                if i in used: continue
                if ((p.bbox_w <= ow+EPS and p.bbox_h <= oh+EPS) or
                    (p.bbox_h <= ow+EPS and p.bbox_w <= oh+EPS)):
                    p.reused_from_offcut = True; used.add(i); reused += 1; break
        return reused


# ============================================================
# SCORING
# ============================================================

class LayoutScorer:

    @staticmethod
    def symmetry_error(pieces: list, room: RoomSpec, tile: TileFormat) -> float:
        hw, hh = tile.width/2, tile.height/2
        minx, miny, maxx, maxy = room.polygon().bounds
        left, right, bottom, top = [], [], [], []
        for p in pieces:
            if not p.is_cut: continue
            px0, py0, px1, py1 = p.polygon.bounds
            if px0-minx <= hw: left.append(p.bbox_w)
            if maxx-px1 <= hw: right.append(p.bbox_w)
            if py0-miny <= hh: bottom.append(p.bbox_h)
            if maxy-py1 <= hh: top.append(p.bbox_h)
        def avg(v): return float(np.mean(v)) if v else 0.0
        return abs(avg(left)-avg(right)) + abs(avg(bottom)-avg(top))

    @staticmethod
    def tiles_to_purchase(pieces: list, tile: TileFormat) -> int:
        return ceil(sum(p.area_m2 for p in pieces) / tile.area_m2 * (1+WASTE_FACTOR))

    @staticmethod
    def calculate(result: LayoutResult) -> LayoutScore:
        pieces  = result.pieces
        covered = sum(p.area_m2 for p in pieces)
        n_buy   = LayoutScorer.tiles_to_purchase(pieces, result.tile)
        src_m2  = n_buy * result.tile.area_m2
        waste   = max(src_m2 - covered, 0.0)
        wpct    = waste/src_m2*100 if src_m2 > EPS else 0.0
        full    = sum(1 for p in pieces if p.is_full)
        cuts    = sum(1 for p in pieces if p.is_cut)
        smalls  = sum(1 for p in pieces if p.is_small_cut)
        reuse   = sum(1 for p in pieces if p.reused_from_offcut)
        sym     = LayoutScorer.symmetry_error(pieces, result.room, result.tile)
        score   = wpct*2.5 + smalls*40 + cuts*1.8 + sym/25 - full*0.8 - reuse*5
        g = result.params.optimize_goal
        if g == OptimizeGoal.NO_SMALL_CUTS: score += smalls*80
        elif g == OptimizeGoal.MIN_WASTE:   score += wpct*4
        elif g == OptimizeGoal.SYMMETRY:    score += sym/8
        elif g == OptimizeGoal.CUT_REUSE:   score -= reuse*15
        return LayoutScore(waste_m2=round(waste,4), waste_pct=round(wpct,2),
                           full_tiles=full, cut_pieces=cuts, small_cuts=smalls,
                           symmetry_error=round(sym,2), reusable_pairs=reuse,
                           score=round(score,3), tiles_to_purchase=n_buy)


# ============================================================
# LAYOUT ENGINE
# ============================================================

class LayoutEngine:
    def generate(self, room: RoomSpec, tile: TileFormat,
                 params: LayoutParams, obstacles: list) -> LayoutResult:
        area    = GeometryEngine.work_area(room, obstacles)
        pattern = PATTERN_FACTORY[params.pattern]()
        pieces  = []; idx = 1

        for src_id, tile_poly in pattern.generate_tiles(room, tile, params):
            clipped = GeometryEngine.clip_to_area(tile_poly, area)
            if clipped is None: continue
            geoms = list(clipped.geoms) if clipped.geom_type == "MultiPolygon" else [clipped]
            for geom in geoms:
                if geom.area < 1: continue
                minx, miny, w, h = GeometryEngine.bbox_size(geom)
                is_full  = GeometryEngine.is_full_piece(geom, tile.width, tile.height)
                is_cut   = not is_full
                is_small = is_cut and min(w,h) < params.small_cut_mm
                is_rect  = GeometryEngine.is_rectangular(geom)
                obs_cut  = any(tile_poly.intersects(o.polygon()) for o in obstacles) if obstacles and is_cut else False
                pieces.append(TilePiece(
                    index=idx, source_tile_id=src_id, polygon=geom,
                    bbox_w=w, bbox_h=h, area_m2=geom.area/1e6,
                    is_full=is_full, is_cut=is_cut, is_small_cut=is_small,
                    is_obstacle_cut=obs_cut, is_rectangular=is_rect,
                    pattern=params.pattern.value, x=minx, y=miny,
                ))
                idx += 1

        pieces = sorted(pieces, key=lambda p: (round(p.y,0), round(p.x,0)))
        for i, p in enumerate(pieces, 1): p.index = i
        CutMapEngine.mark_reuse_candidates(pieces, tile, params.small_cut_mm)

        dummy = LayoutResult(room=room, tile=tile, params=params, pieces=pieces,
                             score=LayoutScore(0,0,0,0,0,0,0,999999,0), obstacles=obstacles)
        dummy.score = LayoutScorer.calculate(dummy)
        return dummy


# ============================================================
# OPTIMIZER
# ============================================================

class LayoutOptimizer:
    def __init__(self): self.engine = LayoutEngine()

    def optimize(self, room, tile, base_params, obstacles,
                 search_step_mm=50, max_candidates=400) -> LayoutResult:
        sx = list(np.arange(0, min(tile.width,  room.bbox_width),  search_step_mm))
        sy = list(np.arange(0, min(tile.height, room.bbox_height), search_step_mm))
        angles  = [45.0] if base_params.pattern == PatternType.DIAGONAL else [base_params.angle_deg]
        offsets = [0.33,0.5,0.66] if base_params.pattern == PatternType.OFFSET else [base_params.offset_ratio]
        candidates = list(itertools.product(sx, sy, offsets, angles))
        if len(candidates) > max_candidates:
            idxs = np.linspace(0, len(candidates)-1, max_candidates).astype(int)
            candidates = [candidates[i] for i in idxs]
        best = None
        bar  = st.progress(0, text=f"Оптимизация: 0/{len(candidates)}")
        for i, (x, y, off, ang) in enumerate(candidates, 1):
            p = LayoutParams(pattern=base_params.pattern, offset_ratio=float(off),
                             angle_deg=float(ang), joint_mm=base_params.joint_mm,
                             start_x=float(x), start_y=float(y),
                             small_cut_mm=base_params.small_cut_mm,
                             optimize_goal=base_params.optimize_goal,
                             modular_sequence=base_params.modular_sequence)
            r = self.engine.generate(room, tile, p, obstacles)
            if best is None or r.score.score < best.score.score: best = r
            bar.progress(i/len(candidates), text=f"Оптимизация: {i}/{len(candidates)}")
        bar.empty()
        return best


# ============================================================
# CAD VISUALIZER
# ============================================================

class CADVisualizer:
    @staticmethod
    def draw(result: LayoutResult, show_full=True, show_cuts=True,
             show_risks=True, show_reuse=True, show_numbers=True,
             show_obstacles=True) -> go.Figure:
        fig  = go.Figure()
        poly = result.room.polygon()
        minx, miny, maxx, maxy = poly.bounds
        bw, bh = maxx-minx, maxy-miny

        rx, ry = poly.exterior.xy
        fig.add_trace(go.Scatter(x=list(rx), y=list(ry), mode="lines", fill="toself",
                                 fillcolor=COLOR_ROOM, line=dict(color=COLOR_TEXT, width=2),
                                 showlegend=False, hoverinfo="skip"))

        for gx in np.arange(minx, maxx+1, 500):
            fig.add_shape(type="line", x0=gx, y0=miny, x1=gx, y1=maxy,
                          line=dict(color=COLOR_GRID, width=0.4), layer="below")
        for gy in np.arange(miny, maxy+1, 500):
            fig.add_shape(type="line", x0=minx, y0=gy, x1=maxx, y1=gy,
                          line=dict(color=COLOR_GRID, width=0.4), layer="below")

        for p in result.pieces:
            if p.is_full and not show_full: continue
            if p.is_cut and p.is_small_cut and not show_risks: continue
            if p.is_cut and not p.is_small_cut and not show_cuts: continue
            if p.reused_from_offcut and not show_reuse: continue
            fill = (COLOR_RISK   if p.is_small_cut else
                    COLOR_REUSED if p.reused_from_offcut else
                    COLOR_CUT    if p.is_cut else COLOR_FULL)
            px, py = p.polygon.exterior.xy
            fig.add_trace(go.Scatter(
                x=list(px), y=list(py), mode="lines", fill="toself",
                fillcolor=fill, line=dict(color=COLOR_GRID, width=1),
                hovertemplate=f"№{p.index} {p.cut_type}<br>{p.size_label}<br>{p.area_m2:.4f} м²<extra></extra>",
                showlegend=False))
            if show_numbers and p.bbox_w >= 100 and p.bbox_h >= 80:
                fig.add_annotation(x=p.x+p.bbox_w/2, y=p.y+p.bbox_h/2,
                    text=f"<b>{p.index}</b><br>{p.size_label}",
                    showarrow=False, font=dict(color=COLOR_TEXT, size=9))

        if show_obstacles:
            for obs in result.obstacles:
                fig.add_shape(type="rect", x0=obs.x, y0=obs.y,
                    x1=obs.x+obs.width, y1=obs.y+obs.height,
                    line=dict(color=COLOR_ACCENT, width=2), fillcolor=COLOR_DOOR)
                fig.add_annotation(x=obs.x+obs.width/2, y=obs.y+obs.height/2,
                    text=obs.name, showarrow=False, font=dict(color=COLOR_ACCENT, size=11))

        fig.update_layout(
            title=f"{result.room.name} · {result.params.pattern.value} · score {result.score.score}",
            paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG, font=dict(color=COLOR_TEXT),
            height=700, margin=dict(l=20,r=20,t=50,b=20),
            xaxis=dict(visible=True, gridcolor=COLOR_GRID, zeroline=False,
                       scaleanchor="y", scaleratio=1, title="X, мм"),
            yaxis=dict(visible=True, gridcolor=COLOR_GRID, zeroline=False, title="Y, мм"),
        )
        fig.update_xaxes(range=[minx-bw*0.05, maxx+bw*0.05])
        fig.update_yaxes(range=[miny-bh*0.05, maxy+bh*0.05])
        return fig


# ============================================================
# PRINT MAP EXPORTER  (PDF A4/A3 landscape, print-ready)
# ============================================================

class PrintMapExporter:
    @staticmethod
    def export(result: LayoutResult, page_size: str = "A4") -> bytes:
        try:
            from reportlab.lib.pagesizes import A4, A3, landscape
            from reportlab.lib.units import mm
            from reportlab.pdfgen import canvas as pdf_canvas
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
        except Exception as exc:
            raise RuntimeError(f"pip install reportlab: {exc}")

        font = "Helvetica"
        for fname, fpath in [("DejaVuSans","/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
                              ("Arial","C:/Windows/Fonts/arial.ttf")]:
            try:
                pdfmetrics.registerFont(TTFont(fname, fpath)); font = fname; break
            except Exception: pass

        pw, ph  = landscape(A3 if page_size == "A3" else A4)
        MARGIN  = 15*mm
        LEGEND_H= 28*mm
        HEADER_H= 20*mm
        draw_w  = pw - 2*MARGIN
        draw_h  = ph - 2*MARGIN - LEGEND_H - HEADER_H

        poly = result.room.polygon()
        rminx, rminy, rmaxx, rmaxy = poly.bounds
        room_w_mm = rmaxx - rminx
        room_h_mm = rmaxy - rminy
        scale = min(draw_w / room_w_mm, draw_h / room_h_mm)

        ox = MARGIN
        oy = MARGIN + LEGEND_H

        def tx(x): return ox + (x - rminx) * scale
        def ty(y): return oy + (y - rminy) * scale

        buf = BytesIO()
        c   = pdf_canvas.Canvas(buf, pagesize=(pw, ph))

        # ── Header ─────────────────────────────────────────
        c.setFont(font, 13)
        c.setFillColorRGB(0.05, 0.07, 0.12)
        c.drawString(MARGIN, ph - MARGIN - 10*mm,
            f"TileCut  |  {result.room.name}  |  "
            f"Плитка {result.tile.width:.0f}x{result.tile.height:.0f} мм  |  "
            f"{result.params.pattern.value}")
        c.setFont(font, 9)
        c.drawString(MARGIN, ph - MARGIN - 17*mm,
            f"Площадь: {result.room.area_m2:.2f} м²   "
            f"Плиток купить: {result.score.tiles_to_purchase}   "
            f"Отход: {result.score.waste_pct:.1f}%   "
            f"Мелких подрезок: {result.score.small_cuts}   "
            f"Score: {result.score.score}")

        # ── Room polygon ────────────────────────────────────
        pts = [(tx(x), ty(y)) for x,y in poly.exterior.coords]
        path = c.beginPath()
        path.moveTo(*pts[0])
        for pt in pts[1:]: path.lineTo(*pt)
        path.close()
        c.setFillColorRGB(0.97, 0.97, 0.95)
        c.setStrokeColorRGB(0.1, 0.1, 0.1)
        c.setLineWidth(1.5)
        c.drawPath(path, stroke=1, fill=1)

        # ── Tiles ───────────────────────────────────────────
        CLR = {"full":(0.82,0.82,0.78),"cut":(0.95,0.68,0.15),
               "risk":(0.90,0.28,0.22),"reuse":(0.18,0.78,0.44)}

        for p in result.pieces:
            coords = list(p.polygon.exterior.coords)
            pts_t  = [(tx(x), ty(y)) for x,y in coords]
            rgb    = (CLR["risk"]  if p.is_small_cut else
                      CLR["reuse"] if p.reused_from_offcut else
                      CLR["cut"]   if p.is_cut else CLR["full"])
            path = c.beginPath()
            path.moveTo(*pts_t[0])
            for pt in pts_t[1:]: path.lineTo(*pt)
            path.close()
            c.setFillColorRGB(*rgb)
            c.setStrokeColorRGB(0.35, 0.40, 0.50)
            c.setLineWidth(0.4)
            c.drawPath(path, stroke=1, fill=1)

            pw_px = p.bbox_w * scale
            ph_px = p.bbox_h * scale
            if pw_px > 14 and ph_px > 10:
                cx = tx(p.x + p.bbox_w/2)
                cy = ty(p.y + p.bbox_h/2)
                c.setFont(font, max(5, min(8, int(pw_px/5))))
                c.setFillColorRGB(0.05, 0.05, 0.05)
                c.drawCentredString(cx, cy, str(p.index))

        # ── Obstacles ───────────────────────────────────────
        for obs in result.obstacles:
            c.setFillColorRGB(0.32, 0.22, 0.10)
            c.setStrokeColorRGB(0.9, 0.63, 0.12)
            c.setLineWidth(1.2)
            c.rect(tx(obs.x), ty(obs.y), obs.width*scale, obs.height*scale, stroke=1, fill=1)
            c.setFont(font, 7); c.setFillColorRGB(1,1,1)
            c.drawCentredString(tx(obs.x+obs.width/2), ty(obs.y+obs.height/2), obs.name)

        # ── Dimensions ──────────────────────────────────────
        c.setStrokeColorRGB(0.1, 0.1, 0.5)
        c.setFillColorRGB(0.1, 0.1, 0.5)
        c.setLineWidth(0.6); c.setFont(font, 7)
        yd = oy - 6*mm
        c.line(tx(rminx), yd, tx(rmaxx), yd)
        c.drawCentredString((tx(rminx)+tx(rmaxx))/2, yd-4*mm, f"{room_w_mm:.0f} мм")
        xd = ox - 6*mm
        c.line(xd, ty(rminy), xd, ty(rmaxy))
        c.drawCentredString(xd-5*mm, (ty(rminy)+ty(rmaxy))/2, f"{room_h_mm:.0f} мм")

        # ── Legend ──────────────────────────────────────────
        ly = MARGIN + 4*mm
        items = [((0.82,0.82,0.78),"Целая плитка"),((0.95,0.68,0.15),"Подрезка"),
                 ((0.90,0.28,0.22),"Мелкая (риск)"),((0.18,0.78,0.44),"Reuse остаток")]
        lx = MARGIN
        for rgb, label in items:
            c.setFillColorRGB(*rgb); c.setStrokeColorRGB(0.3,0.3,0.3)
            c.rect(lx, ly, 12*mm, 8*mm, stroke=1, fill=1)
            c.setFillColorRGB(0,0,0); c.setFont(font, 8)
            c.drawString(lx+14*mm, ly+2*mm, label)
            lx += 55*mm

        c.save()
        return buf.getvalue()


# ============================================================
# EXPORTERS
# ============================================================

class ExcelExporter:
    @staticmethod
    def export(results: list, purchase: dict = None) -> bytes:
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            for i, r in enumerate(results):
                sname = (r.room.name or f"Room{i+1}")[:20]
                r.to_dataframe().to_excel(writer, index=False, sheet_name=f"{sname}_cut")
                CutMapEngine.estimate_offcuts(r).to_excel(writer, index=False, sheet_name=f"{sname}_off")
            pd.DataFrame([{
                "Комната": r.room.name, "Площадь м²": round(r.room.area_m2,2),
                "Плитка": f"{r.tile.width:.0f}x{r.tile.height:.0f}",
                "Pattern": r.params.pattern.value, "Score": r.score.score,
                "Waste %": r.score.waste_pct, "Waste м²": r.score.waste_m2,
                "Плиток": r.score.tiles_to_purchase, "Мелких": r.score.small_cuts,
                "Reuse": r.score.reusable_pairs,
            } for r in results]).to_excel(writer, index=False, sheet_name="Summary")
            if purchase:
                pd.DataFrame([purchase]).to_excel(writer, index=False, sheet_name="Смета")
        return buf.getvalue()


class SVGExporter:
    @staticmethod
    def export(result: LayoutResult) -> bytes:
        _, _, bw, bh = GeometryEngine.bbox_size(result.room.polygon())
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {bw} {bh}">',
            '<style>text{font-family:Arial;font-size:28px;fill:#111}'
            '.full{fill:#ddd;stroke:#333;stroke-width:2}'
            '.cut{fill:#f2b84b;stroke:#333;stroke-width:2}'
            '.risk{fill:#e74c3c;stroke:#333;stroke-width:2}'
            '.reuse{fill:#2ecc71;stroke:#333;stroke-width:2}'
            '.obs{fill:#654321;stroke:#e8a020;stroke-width:4}</style>',
        ]
        for p in result.pieces:
            pts = " ".join(f"{x:.1f},{bh-y:.1f}" for x,y in p.polygon.exterior.coords)
            cls = ("risk" if p.is_small_cut else "reuse" if p.reused_from_offcut
                   else "cut" if p.is_cut else "full")
            parts.append(f'<polygon class="{cls}" points="{pts}"/>')
            if p.bbox_w >= 120 and p.bbox_h >= 120:
                parts.append(f'<text x="{p.x+p.bbox_w/2:.1f}" '
                              f'y="{bh-(p.y+p.bbox_h/2):.1f}" text-anchor="middle">{p.index}</text>')
        for obs in result.obstacles:
            parts.append(f'<rect class="obs" x="{obs.x}" y="{bh-obs.y-obs.height}" '
                         f'width="{obs.width}" height="{obs.height}"/>')
        parts.append("</svg>")
        return "\n".join(parts).encode("utf-8")


class DXFExporter:
    @staticmethod
    def export(result: LayoutResult) -> bytes:
        try: import ezdxf
        except Exception as exc: raise RuntimeError(f"pip install ezdxf: {exc}")
        doc = ezdxf.new("R2010"); msp = doc.modelspace()
        for name, color in [("FULL_TILES",8),("CUT_TILES",30),
                             ("RISK_CUTS",1),("REUSED_CUTS",3),("OBSTACLES",2),("TEXT",7)]:
            doc.layers.add(name, color=color)
        for p in result.pieces:
            layer = ("RISK_CUTS" if p.is_small_cut else "REUSED_CUTS" if p.reused_from_offcut
                     else "CUT_TILES" if p.is_cut else "FULL_TILES")
            msp.add_lwpolyline(list(p.polygon.exterior.coords), close=True, dxfattribs={"layer": layer})
            if p.bbox_w >= 120 and p.bbox_h >= 120:
                msp.add_text(str(p.index), dxfattribs={"height":70,"layer":"TEXT"}).set_placement(
                    (p.x+p.bbox_w/2, p.y+p.bbox_h/2))
        for obs in result.obstacles:
            msp.add_lwpolyline(
                [(obs.x,obs.y),(obs.x+obs.width,obs.y),
                 (obs.x+obs.width,obs.y+obs.height),(obs.x,obs.y+obs.height)],
                close=True, dxfattribs={"layer":"OBSTACLES"})
        fd, path = tempfile.mkstemp(suffix=".dxf")
        os.close(fd)
        try:
            doc.saveas(path)
            with open(path,"rb") as f: return f.read()
        finally:
            os.unlink(path)


# ============================================================
# STREAMLIT UI
# ============================================================

def configure_page():
    st.markdown(f"""<style>
    .stApp{{background:{COLOR_BG};color:{COLOR_TEXT};}}
    [data-testid="stSidebar"]{{background:{COLOR_PANEL};}}
    </style>""", unsafe_allow_html=True)


def room_ui(prefix: str, idx: int):
    """UI одной комнаты. Возвращает (room, tile, params, obstacles, meta)."""
    with st.sidebar.expander(f"🏠 Комната {idx+1}", expanded=(idx == 0)):

        room_name = st.text_input("Название", f"Комната {idx+1}", key=f"{prefix}_name")

        shape_label = st.radio("Форма", ["Прямоугольник","Произвольный полигон"],
                               key=f"{prefix}_shape", horizontal=True)

        custom_points = []
        if shape_label == "Прямоугольник":
            room_w = st.number_input("Ширина, мм", min_value=100.0, value=4200.0, step=10.0, key=f"{prefix}_w")
            room_h = st.number_input("Высота, мм", min_value=100.0, value=3100.0, step=10.0, key=f"{prefix}_h")
            shape_type = RoomShapeType.RECTANGLE
        else:
            shape_type = RoomShapeType.CUSTOM; room_w = room_h = 0.0
            st.caption("Точки X Y (мм), по одной на строке. По часовой стрелке.")
            st.caption("Г-образная — 6 точек, скос — 5 точек.")
            raw = st.text_area("Точки полигона", key=f"{prefix}_pts", height=130,
                value="0 0\n4200 0\n4200 2000\n2000 2000\n2000 3100\n0 3100")
            for line in raw.strip().splitlines():
                parts = line.strip().split()
                if len(parts) == 2:
                    try: custom_points.append((float(parts[0]), float(parts[1])))
                    except ValueError: pass
            if len(custom_points) < 3:
                st.error("Минимум 3 точки.")

        presets = {"600x1200":(600.,1200.),"600x600":(600.,600.),"300x600":(300.,600.),
                   "750x1500":(750.,1500.),"800x1600":(800.,1600.),"1200x1200":(1200.,1200.),"Свой":None}
        preset = st.selectbox("Формат плитки", list(presets.keys()), key=f"{prefix}_preset")
        dw, dh = presets[preset] if presets[preset] else (600., 1200.)
        tile_w    = st.number_input("Ширина плитки, мм", 50.0, value=dw, step=10.0, key=f"{prefix}_tw")
        tile_h    = st.number_input("Высота плитки, мм", 50.0, value=dh, step=10.0, key=f"{prefix}_th")
        joint     = st.number_input("Шов, мм", 0.0, value=2.0, step=0.5, key=f"{prefix}_j")
        small_cut = st.number_input("Мелкая подрезка < мм", 10.0, value=100.0, step=10.0, key=f"{prefix}_sc")

        pat_label = st.selectbox("Раскладка",
            ["grid — прямая","offset — смещение","diagonal — диагональ 45°","modular — модульная"],
            key=f"{prefix}_pat")
        pattern = PatternType(pat_label.split(" — ")[0])
        offset_ratio = st.select_slider("Смещение ряда", [0.25,0.33,0.5,0.66], value=0.5, key=f"{prefix}_off")
        angle    = 45.0 if pattern == PatternType.DIAGONAL else st.number_input("Угол °", value=0.0, step=1.0, key=f"{prefix}_ang")
        start_x  = st.number_input("Старт X, мм", value=0.0, step=10.0, key=f"{prefix}_sx")
        start_y  = st.number_input("Старт Y, мм", value=0.0, step=10.0, key=f"{prefix}_sy")

        goal_map = {"Без узких полос":OptimizeGoal.NO_SMALL_CUTS,
                    "Минимум отхода":OptimizeGoal.MIN_WASTE,
                    "Симметрия":OptimizeGoal.SYMMETRY,
                    "Reuse остатков":OptimizeGoal.CUT_REUSE}
        goal_label  = st.selectbox("Цель оптимизации", list(goal_map.keys()), key=f"{prefix}_goal")
        search_step = st.number_input("Шаг оптимизатора, мм", 5, 200, value=50, step=5, key=f"{prefix}_step")
        run_opt     = st.button(f"▶ Оптимизировать {idx+1}", key=f"{prefix}_opt", use_container_width=True)

        n_obs = st.number_input("Проёмов", 0, 6, value=0, step=1, key=f"{prefix}_nobs")
        obstacles = []
        for k in range(int(n_obs)):
            st.markdown(f"**Проём {k+1}**")
            obstacles.append(Obstacle(
                name=st.text_input("Название", f"Проём {k+1}", key=f"{prefix}_oname{k}"),
                x   =st.number_input("X, мм",      0.0, value=300.0, step=10.0, key=f"{prefix}_ox{k}"),
                y   =st.number_input("Y, мм",      0.0, value=0.0,   step=10.0, key=f"{prefix}_oy{k}"),
                width =st.number_input("Ширина, мм",1.0, value=800.0, step=10.0, key=f"{prefix}_ow{k}"),
                height=st.number_input("Высота, мм",1.0, value=200.0, step=10.0, key=f"{prefix}_oh{k}"),
            ))

        layers = {
            "show_full":      st.checkbox("Целые",     True,  key=f"{prefix}_lf"),
            "show_cuts":      st.checkbox("Подрезки",  True,  key=f"{prefix}_lc"),
            "show_risks":     st.checkbox("Мелкие",    True,  key=f"{prefix}_lr"),
            "show_reuse":     st.checkbox("Reuse",     True,  key=f"{prefix}_lu"),
            "show_numbers":   st.checkbox("Номера",    True,  key=f"{prefix}_ln"),
            "show_obstacles": st.checkbox("Проёмы",    True,  key=f"{prefix}_lo"),
        }

    room   = RoomSpec(name=room_name, width=room_w, height=room_h,
                      shape_type=shape_type, custom_points=custom_points)
    tile   = TileFormat(width=tile_w, height=tile_h, name=preset)
    params = LayoutParams(pattern=pattern, offset_ratio=offset_ratio, angle_deg=angle,
                          joint_mm=joint, start_x=start_x, start_y=start_y,
                          small_cut_mm=small_cut, optimize_goal=goal_map[goal_label])
    meta   = {"run_opt": run_opt, "search_step": int(search_step), "layers": layers}
    return room, tile, params, obstacles, meta


def render_purchase_sidebar() -> PurchaseCalc:
    with st.sidebar.expander("💰 Калькулятор закупки", expanded=False):
        cur  = st.selectbox("Валюта", ["грн","USD","EUR","₸"], index=0)
        tp   = st.number_input(f"Плитка за м² ({cur})", 0.0, value=0.0, step=50.0)
        tpp  = st.number_input("Штук в упаковке",       1.0, value=4.0,  step=1.0)
        pa   = st.number_input("м² в упаковке",         0.01, value=1.44, step=0.01)
        adh  = st.number_input(f"Клей ({cur}/кг)",      0.0, value=0.0,  step=5.0)
        grout= st.number_input(f"Затирка ({cur}/кг)",   0.0, value=0.0,  step=5.0)
        under= st.number_input(f"Гидроизоляция ({cur}/м²)", 0.0, value=0.0, step=10.0)
    return PurchaseCalc(tile_price_per_m2=tp, tiles_per_pack=tpp, pack_area_m2=pa,
                        adhesive_price_per_kg=adh, grout_price_per_kg=grout,
                        underlay_price_per_m2=under, currency=cur)


def render_room_section(result: LayoutResult, layers: dict, room_idx: int):
    s = result.score
    c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
    c1.metric("Score ↓",       s.score)
    c2.metric("Площадь м²",    f"{result.room.area_m2:.2f}")
    c3.metric("Отход %",       f"{s.waste_pct:.1f}%")
    c4.metric("Плиток купить", s.tiles_to_purchase)
    c5.metric("Подрезок",      s.cut_pieces)
    c6.metric("Мелких",        s.small_cuts)
    c7.metric("Reuse",         s.reusable_pairs)

    t1,t2,t3,t4,t5 = st.tabs(["2D CAD","Cut map","Offcuts","Scoring","Замечания"])

    with t1: st.plotly_chart(CADVisualizer.draw(result, **layers), use_container_width=True)

    with t2:
        df = result.to_dataframe()
        st.caption(f"Деталей: {len(df)}")
        st.dataframe(df, use_container_width=True, height=460)

    with t3:
        off = CutMapEngine.estimate_offcuts(result)
        if off.empty: st.info("Прямоугольных остатков нет.")
        else:
            st.caption(f"Остатков: {len(off)}")
            st.dataframe(off, use_container_width=True, height=380)

    with t4:
        st.dataframe(pd.DataFrame([{
            "Waste м²":s.waste_m2, "Waste %":s.waste_pct,
            "Плиток":s.tiles_to_purchase, "Full":s.full_tiles,
            "Cuts":s.cut_pieces, "Small":s.small_cuts,
            "Sym.err":s.symmetry_error, "Reuse":s.reusable_pairs, "Score":s.score,
        }]), use_container_width=True)

    with t5:
        notes = [
            f"{'⚠️' if s.small_cuts else '✅'} Мелких подрезок: {s.small_cuts}",
            (f"⚠️ Отход {s.waste_pct:.1f}% — высокий." if s.waste_pct > 20 else
             f"〰️ Отход {s.waste_pct:.1f}% — терпимо." if s.waste_pct > 12 else
             f"✅ Отход {s.waste_pct:.1f}% — норма."),
        ]
        if s.reusable_pairs: notes.append(f"♻️ Reuse кандидатов: {s.reusable_pairs}")
        nr = sum(1 for p in result.pieces if p.is_cut and not p.is_rectangular)
        if nr: notes.append(f"⚠️ L-образных подрезок (проёмы): {nr} — в отход.")
        if result.room.shape_type == RoomShapeType.CUSTOM:
            notes.append("ℹ️ Нестандартный полигон: проверь точки на CAD-виде.")
        notes.append(f"{'⚠️' if s.symmetry_error > 50 else '✅'} Симметрия краёв: {s.symmetry_error:.0f} мм")
        for n in notes: st.write(n)

    # Экспорт комнаты
    st.markdown("**Экспорт комнаты**")
    e1,e2,e3,e4,e5 = st.columns(5)
    with e1:
        ps = st.selectbox("Размер PDF", ["A4","A3"], key=f"ps_{room_idx}", label_visibility="collapsed")
    with e2:
        try:
            st.download_button("🖨 Карта PDF",
                data=PrintMapExporter.export(result, ps),
                file_name=f"map_{result.room.name}.pdf",
                mime="application/pdf", use_container_width=True, key=f"pmap_{room_idx}")
        except Exception as ex: st.caption(f"PDF: {ex}")
    with e3:
        st.download_button("SVG", data=SVGExporter.export(result),
            file_name=f"layout_{result.room.name}.svg",
            mime="image/svg+xml", use_container_width=True, key=f"svg_{room_idx}")
    with e4:
        try:
            st.download_button("DXF", data=DXFExporter.export(result),
                file_name=f"layout_{result.room.name}.dxf",
                mime="application/dxf", use_container_width=True, key=f"dxf_{room_idx}")
        except Exception as ex: st.caption(f"DXF: {ex}")


def render_project_summary(results: list, calc: PurchaseCalc):
    st.markdown("---")
    st.subheader("📊 Сводка проекта")

    cols = st.columns(len(results)+1)
    total_tiles = sum(r.score.tiles_to_purchase for r in results)
    total_area  = sum(r.room.area_m2 for r in results)
    for i, r in enumerate(results):
        cols[i].metric(r.room.name, f"{r.room.area_m2:.2f} м²", f"{r.score.tiles_to_purchase} шт")
    cols[-1].metric("ИТОГО", f"{total_area:.2f} м²", f"{total_tiles} шт")

    st.subheader("🧾 Смета закупки")
    purchase = calc.calculate(results)
    st.dataframe(pd.DataFrame([{"Статья":k,"Значение":v} for k,v in purchase.items()]),
                 use_container_width=True, hide_index=True)

    st.markdown("**Экспорт проекта**")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("📥 Excel — все комнаты + смета",
            data=ExcelExporter.export(results, purchase),
            file_name="tilecut_project.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True)
    with col2:
        # Объединённый PDF всех карт (требует pypdf)
        try:
            from pypdf import PdfWriter, PdfReader
            writer = PdfWriter()
            for r in results:
                for page in PdfReader(BytesIO(PrintMapExporter.export(r))).pages:
                    writer.add_page(page)
            buf = BytesIO(); writer.write(buf)
            st.download_button("🖨 Все карты (PDF)",
                data=buf.getvalue(), file_name="tilecut_all_maps.pdf",
                mime="application/pdf", use_container_width=True)
        except ImportError:
            st.caption("pip install pypdf — для объединённого PDF всех комнат")
        except Exception as ex:
            st.caption(f"PDF: {ex}")


# ============================================================
# MAIN
# ============================================================

def main():
    try:
        st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="🧱")
    except Exception:
        pass
    configure_page()

    if not SHAPELY_AVAILABLE:
        st.error(f"pip install shapely\n\n{SHAPELY_ERROR}")
        st.stop()

    st.title("🧱 TileCut Engineering System v3")
    st.caption("Многокомнатность · Произвольные формы · Печатная карта · Смета закупки")

    st.sidebar.title("🧱 TileCut v3")
    n_rooms = int(st.sidebar.number_input("Количество комнат", 1, 8, value=1, step=1))
    calc    = render_purchase_sidebar()

    engine    = LayoutEngine()
    optimizer = LayoutOptimizer()
    results   = []

    for i in range(n_rooms):
        room, tile, params, obstacles, meta = room_ui(f"r{i}", i)

        if room.shape_type == RoomShapeType.CUSTOM and len(room.custom_points) < 3:
            st.warning(f"Комната {i+1}: задай минимум 3 точки полигона.")
            continue

        with st.spinner(f"Расчёт комнаты {i+1}…"):
            if meta["run_opt"]:
                result = optimizer.optimize(room=room, tile=tile, base_params=params,
                                            obstacles=obstacles,
                                            search_step_mm=meta["search_step"])
                st.success(f"Комната {i+1} оптимизирована: score={result.score.score}, "
                           f"start=({result.params.start_x:.0f}, {result.params.start_y:.0f})")
            else:
                result = engine.generate(room, tile, params, obstacles)

        results.append(result)
        st.markdown(f"### 🏠 {room.name}")
        render_room_section(result, meta["layers"], i)

    if results:
        render_project_summary(results, calc)


if __name__ == "__main__":
    main()