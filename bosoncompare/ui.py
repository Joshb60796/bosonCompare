"""Phosphor instrument console for three Boson cubes.

The time-series plot carries two vertical lines. They start on the first
and last frame and drag along the reference clock (S1 when that cube is
loaded). The indexes they select are applied to every cube for peak-to-peak,
the other temporal maps, and principal component 2.
"""

from __future__ import annotations

import gc
import os
import sys

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QSize, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from bosoncompare.analysis import (
    BOX,
    DISPLAY,
    FPS,
    HEAT_OFF,
    HEAT_ON,
    DIVERGING_MODES,
    MODE_LABEL,
    MODES,
    N_FRAMES,
    NUDGE,
    RANGE_MODES,
    SLIDER_MODES,
    SMOOTH,
    T0_S,
    TLINEAR,
    display_image,
    extract_trace,
    left_axis_is_difference,
    load_log,
    mode_caption,
    pad_limits,
    percentile_limits,
    symmetric_limit,
)
from bosoncompare.synthetic import ensure

pg.setConfigOptions(imageAxisOrder="row-major", antialias=True, background="#070b10",
                    foreground="#9eb4c2")


def _quiet_pyqtgraph_app():
    """Skip pyqtgraph's color-scheme hook when a Qt application already exists.

    PySide6 prints ``QStyleHints: unique connections require a pointer to
    member function`` for that hook, and pyqtgraph retries it once per plot
    panel. The console does not use the hook.
    """
    import pyqtgraph.Qt as pgqt
    import pyqtgraph.widgets.GraphicsLayoutWidget as layout_widget

    if getattr(pgqt.mkQApp, "_boson_quiet", False):
        return
    original = pgqt.mkQApp

    def mkQApp(name=None):
        app = QApplication.instance()
        if app is None:
            app = original(name)
        pgqt.QAPP = app
        try:
            dark = app.styleHints().colorScheme() == Qt.ColorScheme.Dark
            app.setProperty("darkMode", dark)
        except AttributeError:
            pass
        if name is not None:
            app.setApplicationName(name)
        return app

    mkQApp._boson_quiet = True
    pgqt.mkQApp = mkQApp
    pg.mkQApp = mkQApp
    layout_widget.mkQApp = mkQApp

N_SETS = 3
SET_NAMES = ("S1", "S2", "S3")
LOC_NAMES = ("A", "B")
LOC_SYMBOLS = ("+", "x")
DIFF_PAIRS = ((0, 1), (0, 2), (1, 2))
# Empty choice means Auto: inferno, or CET-D1 for a signed mode or the difference.
MAP_CHOICES = (
    ("Auto", ""),
    ("inferno", "inferno"),
    ("magma", "magma"),
    ("plasma", "plasma"),
    ("viridis", "viridis"),
    ("cividis", "cividis"),
    ("turbo", "turbo"),
    ("gray", "CET-L1"),
    ("CET-D1", "CET-D1"),
    ("CET-D3", "CET-D3"),
    ("CET-D8", "CET-D8"),
    ("CET-D13", "CET-D13"),
    ("CET-CBD1", "CET-CBD1"),
)
AUTO_SEQUENTIAL = "inferno"
AUTO_DIVERGING = "CET-D1"
# Guide lines on the time plot. Trace colors stay away from both of these.
LINE_GREEN = "#5ef0d0"
LINE_ORANGE = "#ffb15a"
# S1 A, S1 B, S2 A, S2 B, S3 A, S3 B, then S1 A−B, S2 A−B, S3 A−B.
# Red, blue, green, yellow, violet, white, magenta, teal, brown.
# Yellow stays lemon so it does not read as the orange guide, and teal stays
# dark so it does not read as the mint guide.
SAMPLE_COLORS = (
    "#ff2d2d",
    "#245bff",
    "#00b34a",
    "#fff200",
    "#b020f0",
    "#f7f7f2",
    "#ff2bd6",
    "#0e7c86",
    "#8a4518",
)
THERM_COLORS = ("#ff7043", "#5c6bc0", "#8d6e63")


def sample_color(set_index, kind):
    if kind == "AB":
        return SAMPLE_COLORS[6 + set_index]
    loc = 0 if kind == "A" else 1
    return SAMPLE_COLORS[set_index * 2 + loc]

STYLE = """
QWidget {
    background: #070b10;
    color: #d7e4ec;
    font-family: Bahnschrift, "Segoe UI";
    font-size: 13px;
}
QMainWindow { background: #070b10; }
QLabel#wordmark {
    font-family: Bahnschrift, "Segoe UI";
    font-size: 20px;
    letter-spacing: 0.28em;
    color: #e9f7f4;
    font-weight: 600;
}
QLabel#submark {
    color: #6f8b86;
    font-family: "Cascadia Mono", Consolas, monospace;
    font-size: 11px;
    letter-spacing: 0.14em;
}
QLabel#readout {
    color: #5ef0d0;
    font-family: "Cascadia Mono", Consolas, monospace;
    font-size: 12px;
}
QLabel#hint {
    color: #6d8290;
    font-family: "Cascadia Mono", Consolas, monospace;
    font-size: 11px;
}
QFrame#rule { background: #5ef0d0; border: none; max-height: 2px; }
QFrame#panel {
    background: #0c1218;
    border: 1px solid #1a2c38;
    border-radius: 2px;
}
QPushButton {
    background: #101820;
    color: #d7e4ec;
    border: 1px solid #243846;
    border-radius: 2px;
    padding: 6px 8px;
    letter-spacing: 0.06em;
    text-align: left;
}
QPushButton:hover { border-color: #5ef0d0; color: #f4fffc; }
QPushButton:checked {
    background: #102824;
    color: #d8fff6;
    border: 1px solid #5ef0d0;
}
QPushButton#mode {
    padding: 4px 7px;
    letter-spacing: 0.03em;
    font-size: 12px;
}
QPushButton#file { font-family: "Cascadia Mono", Consolas, monospace; font-size: 11px; }
QComboBox, QSpinBox, QDoubleSpinBox {
    background: #101820;
    color: #d7e4ec;
    border: 1px solid #243846;
    border-radius: 2px;
    padding: 3px 8px;
    min-height: 22px;
    font-family: "Cascadia Mono", Consolas, monospace;
}
QComboBox::drop-down { border: none; width: 16px; }
QComboBox QAbstractItemView {
    background: #101820;
    color: #d7e4ec;
    selection-background-color: #12342f;
    selection-color: #d8fff6;
}
QCheckBox { spacing: 6px; color: #c5d5de; }
QCheckBox::indicator {
    width: 14px; height: 14px;
    border: 1px solid #3a5564;
    background: #101820;
}
QCheckBox::indicator:checked { background: #5ef0d0; border-color: #5ef0d0; }
QSlider::groove:horizontal {
    height: 4px;
    background: #15202b;
    border: 1px solid #1e3342;
}
QSlider::handle:horizontal {
    background: #5ef0d0;
    width: 12px;
    margin: -7px 0;
    border-radius: 1px;
}
QStatusBar {
    background: #070b10;
    color: #8aa0b0;
    font-family: "Cascadia Mono", Consolas, monospace;
    font-size: 11px;
}
"""


def short_name(path):
    return os.path.basename(path) if path else "(none)"


class Pane(pg.GraphicsLayoutWidget):
    """A plot well whose size hint stays small so four of them fit the window."""

    def sizeHint(self):
        return QSize(200, 180)

    def minimumSizeHint(self):
        return QSize(80, 80)


class FileSet:
    def __init__(self, index):
        self.index = index
        self.name = SET_NAMES[index]
        self.csv_path = ""
        self.cube_path = ""
        self.t_log = self.t1 = self.t2 = None
        self.cube = None
        self.ny = self.nx = self.nf = 0
        self.t0 = float(T0_S)
        self.display_img = None

    def ready_log(self):
        return self.t_log is not None

    def ready_cube(self):
        return self.cube is not None

    def t_cam(self, fps):
        return self.t0 + np.arange(self.nf) / float(fps)


class MainWindow(QMainWindow):
    def __init__(self, autoload=False):
        _quiet_pyqtgraph_app()
        super().__init__()
        self.setWindowTitle("Boson Compare")
        self.resize(1560, 960)
        self.setStyleSheet(STYLE)

        self.sets = [FileSet(i) for i in range(N_SETS)]
        self.fps = float(FPS)
        self.frame = 0
        self.frame_lo = 0
        self.frame_hi = 0
        self.half = max(0, int(BOX) // 2)
        self.smooth_n = int(SMOOTH)
        self.picks = [None, None]
        self.show_ab = False
        self.subtract_first = False
        self.display_mode = DISPLAY if DISPLAY in MODE_LABEL else "std"
        self.diff_pair = DIFF_PAIRS[0]
        self._last_dir = os.getcwd()
        self._left_auto = True
        self._applying_region = False
        self._range_ready = False

        self._build()
        QApplication.instance().installEventFilter(self)
        if autoload:
            self.load_synthetic(ask=False)

    # ---------------------------------------------------------------- build #
    def _build(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 12, 16, 8)
        root.setSpacing(8)

        header = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(0)
        word = QLabel("BOSON COMPARE")
        word.setObjectName("wordmark")
        sub = QLabel("THREE-CUBE THERMAL CONSOLE")
        sub.setObjectName("submark")
        titles.addWidget(word)
        titles.addWidget(sub)
        header.addLayout(titles)
        header.addStretch(1)
        self.range_readout = QLabel("FRAME RANGE  —")
        self.range_readout.setObjectName("readout")
        header.addWidget(self.range_readout)
        root.addLayout(header)

        rule = QFrame()
        rule.setObjectName("rule")
        rule.setFrameShape(QFrame.NoFrame)
        rule.setFixedHeight(2)
        root.addWidget(rule)

        files = QHBoxLayout()
        files.setSpacing(6)
        self.file_buttons = []
        labels = []
        for name in SET_NAMES:
            labels.extend((f"{name} LOG", f"{name} CUBE"))
        for slot, label in enumerate(labels):
            button = QPushButton(label)
            button.setObjectName("file")
            button.setMinimumHeight(46)
            button.setToolTip("Open a file for this slot")
            button.clicked.connect(self._bind_file(slot))
            files.addWidget(button, 1)
            self.file_buttons.append(button)
        demo = QPushButton("SYNTHETIC")
        demo.setMinimumHeight(46)
        demo.setToolTip(
            "Generate and load the three synthetic uint16 cubes. "
            "Thermistor logs are written beside them and stay closed until opened.")
        demo.clicked.connect(lambda: self.load_synthetic(ask=True, regenerate=True))
        files.addWidget(demo, 1)
        root.addLayout(files)

        modes = QHBoxLayout()
        modes.setSpacing(6)
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_buttons = {}
        tips = {
            "std": "Standard deviation inside the dragged frame range.",
            "var": "Variance inside the dragged frame range.",
            "ptp": "Peak to peak inside the dragged frame range.",
            "mean": "Mean counts inside the dragged frame range.",
            "delta": "Orange line frame minus green line frame.",
            "first": "The single frame chosen with the slider.",
            "pca1": "First principal component over the dragged frame range. "
                    "Each pixel is one sample and time is the variable, with the "
                    "temporal mean removed.",
            "pca2": "Second principal component over the dragged frame range. "
                    "Each pixel is one sample and time is the variable, with the "
                    "temporal mean removed.",
            "pca3": "Third principal component over the dragged frame range. "
                    "Each pixel is one sample and time is the variable, with the "
                    "temporal mean removed.",
            "tsr1": "First logarithmic derivative at the slider frame. "
                    "Fits ln(temperature rise) as a polynomial in ln(time) from the "
                    "green line, the Thermal Wave Imaging reconstruction.",
            "tsr2": "Second logarithmic derivative at the slider frame, from that "
                    "same fit. Thermal Wave Imaging uses this to separate a defect "
                    "from sound material.",
        }
        for key, text in MODES:
            button = QPushButton(text)
            button.setObjectName("mode")
            button.setCheckable(True)
            button.setToolTip(tips[key])
            button.clicked.connect(self._bind_mode(key))
            self.mode_group.addButton(button)
            modes.addWidget(button)
            self.mode_buttons[key] = button
        self.mode_buttons[self.display_mode].setChecked(True)
        modes.addStretch(1)
        modes.addWidget(self._field_label("FPS"))
        self.fps_box = QDoubleSpinBox()
        self.fps_box.setRange(0.1, 1000.0)
        self.fps_box.setDecimals(2)
        self.fps_box.setValue(self.fps)
        self.fps_box.setToolTip("Camera frames per second. Time is frame index divided by this.")
        self.fps_box.valueChanged.connect(self.on_fps)
        modes.addWidget(self.fps_box)
        modes.addWidget(self._field_label("FRAME"))
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, 1)
        self.frame_slider.setValue(0)
        self.frame_slider.setMinimumWidth(180)
        self.frame_slider.setToolTip(
            "Selected frame for FRAME, TSR1, and TSR2. [ and ] step it.")
        self.frame_slider.valueChanged.connect(self.on_frame_slider)
        modes.addWidget(self.frame_slider, 1)
        self.frame_label = QLabel("0")
        self.frame_label.setObjectName("readout")
        self.frame_label.setMinimumWidth(48)
        modes.addWidget(self.frame_label)
        root.addLayout(modes)

        options = QHBoxLayout()
        options.setSpacing(10)
        self.t0_boxes = []
        for index, name in enumerate(SET_NAMES):
            options.addWidget(self._field_label(f"{name} t0"))
            box = QDoubleSpinBox()
            box.setRange(-1000.0, 1000.0)
            box.setDecimals(2)
            box.setSingleStep(0.1)
            box.setValue(0.0)
            box.setSuffix(" s")
            box.setToolTip(self._t0_tip(index))
            box.valueChanged.connect(self._bind_t0(index))
            options.addWidget(box)
            self.t0_boxes.append(box)
        options.addWidget(self._field_label("BOX"))
        self.box_spin = QSpinBox()
        self.box_spin.setRange(1, 15)
        self.box_spin.setSingleStep(2)
        self.box_spin.setValue(self.half * 2 + 1)
        self.box_spin.setToolTip("Odd width of the pixel neighborhood averaged into each trace.")
        self.box_spin.valueChanged.connect(self.on_box)
        options.addWidget(self.box_spin)

        self.ab_box = QCheckBox("A − B trace")
        self.ab_box.setToolTip(
            "Add pixel A minus pixel B for each cube. "
            "With no thermistor log, those traces use the left axis.")
        self.ab_box.toggled.connect(self.on_ab)
        options.addWidget(self.ab_box)
        self.sub_box = QCheckBox("Subtract first frame")
        self.sub_box.setToolTip("Subtract frame 0 before the map and before each trace.")
        self.sub_box.toggled.connect(self.on_subtract)
        options.addWidget(self.sub_box)
        self.left_box = QCheckBox("Left axis: A − B")
        self.left_box.setChecked(True)
        self.left_box.setToolTip(
            "Scale the left axis to the difference between pixels A and B. "
            "Turns on by itself when no thermistor log is loaded.")
        self.left_box.clicked.connect(self.on_left_clicked)
        options.addWidget(self.left_box)
        options.addStretch(1)
        options.addWidget(self._field_label("DIFF"))
        self.diff_combo = QComboBox()
        for a, b in DIFF_PAIRS:
            self.diff_combo.addItem(f"{SET_NAMES[a]} − {SET_NAMES[b]}")
        self.diff_combo.currentIndexChanged.connect(self.on_diff_pair)
        options.addWidget(self.diff_combo)
        root.addLayout(options)

        maps = QHBoxLayout()
        maps.setSpacing(8)
        maps.addWidget(self._field_label("MAP"))
        self.map_combos = []
        for name in (*SET_NAMES, "DIFF"):
            maps.addWidget(self._field_label(name))
            combo = QComboBox()
            combo.setToolTip(self._map_tip(name == "DIFF"))
            for label, data in MAP_CHOICES:
                combo.addItem(label, data)
            combo.currentIndexChanged.connect(self.on_colormap)
            maps.addWidget(combo)
            self.map_combos.append(combo)
        self.diff_map_combo = self.map_combos.pop()
        maps.addStretch(1)
        root.addLayout(maps)

        plots = QHBoxLayout()
        plots.setSpacing(8)
        self.img_plots = []
        self.images = []
        self._marks = []
        for index in range(N_SETS):
            view = self._pane()
            plot = view.addPlot()
            self._style_image_plot(plot, f"{SET_NAMES[index]}  ·  no cube")
            image = pg.ImageItem()
            plot.addItem(image)
            view.scene().sigMouseClicked.connect(
                lambda ev, target=plot: self.on_image_click(target, ev))
            plots.addWidget(view, 1)
            self.img_plots.append(plot)
            self.images.append(image)
        diff_view = self._pane()
        self.diff_plot = diff_view.addPlot()
        self._style_image_plot(self.diff_plot, "S1 − S2")
        self.diff_image = pg.ImageItem()
        self.diff_plot.addItem(self.diff_image)
        plots.addWidget(diff_view, 1)
        # The bars show the shared numbers. Each image keeps its own lookup
        # table, so the bars are not linked to the images.
        self.cbar = pg.ColorBarItem(
            values=(0, 1), width=18, colorMap=AUTO_SEQUENTIAL,
            interactive=False, rounding=0.01)
        plots.addWidget(self._scale_pane(self.cbar), 0)
        self.cbar_diff = pg.ColorBarItem(
            values=(-1, 1), width=18, colorMap=AUTO_DIVERGING,
            interactive=False, rounding=0.01)
        plots.addWidget(self._scale_pane(self.cbar_diff), 0)
        root.addLayout(plots, 5)

        self.series_view = self._pane()
        self.series = self.series_view.addPlot()
        self._style_series(self.series)
        self.right_vb = pg.ViewBox()
        self.series.showAxis("right")
        self.series.scene().addItem(self.right_vb)
        self.series.getAxis("right").linkToView(self.right_vb)
        self.right_vb.setXLink(self.series)
        self.series.vb.sigResized.connect(self._sync_right_view)
        self.legend = self.series.addLegend(offset=(-8, 8), labelTextSize="9pt")
        self.legend.setBrush(pg.mkBrush(7, 11, 16, 180))
        self.legend.setPen(pg.mkPen("#1e3342"))

        self.region = pg.LinearRegionItem(
            values=(0, 1), orientation="vertical", movable=True, swapMode="block",
            brush=pg.mkBrush(94, 240, 208, 28),
            pen=pg.mkPen(LINE_GREEN, width=2),
            hoverBrush=pg.mkBrush(94, 240, 208, 48),
            hoverPen=pg.mkPen("#e9fff8", width=2))
        self.region.lines[0].setPen(pg.mkPen(LINE_GREEN, width=2))
        self.region.lines[1].setPen(pg.mkPen(LINE_ORANGE, width=2))
        self.region.lines[0].setHoverPen(pg.mkPen("#e9fff8", width=2))
        self.region.lines[1].setHoverPen(pg.mkPen("#ffe7c2", width=2))
        self.region.setZValue(8)
        self.series.addItem(self.region)
        self.region.sigRegionChanged.connect(self.on_region_moved)
        self.region.sigRegionChangeFinished.connect(self.on_region_finished)
        self.frame_line = pg.InfiniteLine(
            pos=0, angle=90, movable=False,
            pen=pg.mkPen("#9aa4ad", width=1, style=Qt.PenStyle.DashLine))
        self.frame_line.setZValue(7)
        self.series.addItem(self.frame_line)
        root.addWidget(self.series_view, 4)

        hint = QLabel(
            "left-click sets A, then B    ·    right-click or C clears    ·    "
            "drag the green and orange lines to set the frame range    ·    "
            ", . nudge S1 ±0.1 s    < > nudge S2 ±1.0 s    ; ' nudge S3 ±0.1 s    "
            "[ ] step the frame")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        root.addWidget(hint)
        self.statusBar().showMessage("Open three cubes, or use SYNTHETIC.")
        self._set_axis_titles()

    def _pane(self):
        view = Pane()
        view.setBackground("#070b10")
        view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        view.setMinimumSize(80, 140)
        return view

    def _scale_pane(self, colorbar):
        view = self._pane()
        view.setFixedWidth(84)
        view.setMinimumWidth(84)
        view.addItem(colorbar)
        return view

    def _field_label(self, text):
        label = QLabel(text)
        label.setObjectName("hint")
        return label

    def _t0_tip(self, index):
        keys = (", and . by 0.1 s", "< and > by 1.0 s", "; and ' by 0.1 s")
        return f"{SET_NAMES[index]} camera time of frame 0. Keys {keys[index]}."

    def _style_image_plot(self, plot, title):
        plot.setMenuEnabled(False)
        plot.hideButtons()
        plot.invertY(True)
        plot.setAspectLocked(True)
        plot.showGrid(x=False, y=False)
        plot.setTitle(title, color="#d7e4ec", size="10pt")
        for name in ("left", "bottom"):
            axis = plot.getAxis(name)
            axis.setPen(pg.mkPen("#1e3342"))
            axis.setTextPen(pg.mkPen("#8aa0b0"))
        plot.getAxis("bottom").setLabel("x", units="px")
        plot.getAxis("left").setLabel("y", units="px")
        plot.showAxis("top", False)
        plot.showAxis("right", False)

    def _style_series(self, plot):
        plot.setMenuEnabled(False)
        plot.hideButtons()
        plot.showGrid(x=True, y=True, alpha=0.15)
        plot.setLabel("bottom", "t_rel_s", units="s")
        for name in ("left", "bottom", "right"):
            axis = plot.getAxis(name)
            axis.setPen(pg.mkPen("#1e3342"))
            axis.setTextPen(pg.mkPen("#8aa0b0"))
            axis.enableAutoSIPrefix(False)
        plot.getAxis("left").setPen(pg.mkPen("#5ef0d0"))
        plot.getAxis("right").setPen(pg.mkPen("#ffb15a"))

    def _set_axis_titles(self):
        if self._difference_on_left():
            unit = "°C" if TLINEAR else "counts"
            self.series.setLabel("left", "A minus B", units=unit, color="#5ef0d0")
        else:
            self.series.setLabel("left", "thermistor", units="°C", color="#5ef0d0")
        unit = "°C" if TLINEAR else "counts"
        self.series.getAxis("right").setLabel("camera", units=unit, color="#ffb15a")
        self.series.getAxis("left").enableAutoSIPrefix(False)
        self.series.getAxis("right").enableAutoSIPrefix(False)

    # --------------------------------------------------------------- files #
    def _bind_file(self, slot):
        return lambda: self.on_open_slot(slot)

    def _bind_mode(self, key):
        return lambda: self.on_mode(key)

    def _bind_t0(self, index):
        return lambda value: self.on_t0(index, value)

    def on_open_slot(self, slot):
        index, kind = divmod(slot, 2)
        if kind == 0:
            path, _ = QFileDialog.getOpenFileName(
                self, f"{SET_NAMES[index]} thermistor CSV", self._last_dir,
                "CSV (*.csv);;All files (*.*)")
            if path:
                self._remember(path)
                self._load_csv(index, path)
                self._after_files()
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, f"{SET_NAMES[index]} thermal cube", self._last_dir,
                "NumPy cube (*.npy);;All files (*.*)")
            if path:
                self._remember(path)
                self._load_cube(index, path)
                self._after_files()

    def _remember(self, path):
        folder = os.path.dirname(path)
        if folder:
            self._last_dir = folder

    def _load_csv(self, index, path):
        try:
            t, t1, t2 = load_log(path)
        except (OSError, ValueError, KeyError) as exc:
            QMessageBox.warning(self, "Thermistor log", str(exc))
            return
        fs = self.sets[index]
        fs.csv_path = path
        fs.t_log, fs.t1, fs.t2 = t, t1, t2
        self._label_file(index * 2, f"{fs.name} LOG", path)
        self.statusBar().showMessage(
            f"{fs.name} log {short_name(path)}: {t.size} rows, {t[0]:.2f}–{t[-1]:.2f} s")

    def _load_cube(self, index, path):
        try:
            cube = np.load(path, mmap_mode="r")
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Thermal cube", str(exc))
            return
        if cube.ndim != 3:
            QMessageBox.warning(self, "Thermal cube", f"expected a 3-D cube, got {cube.shape}")
            return
        fs = self.sets[index]
        fs.cube_path = path
        fs.cube = cube
        fs.ny, fs.nx, fs.nf = (int(v) for v in cube.shape)
        self._label_file(index * 2 + 1, f"{fs.name} CUBE", path)
        self._fit_image_view(index)
        self.statusBar().showMessage(
            f"{fs.name} cube {short_name(path)}: {cube.shape} {cube.dtype} "
            f"→ {fs.nf / self.fps:.2f} s at {self.fps:g} frames/s")

    def _label_file(self, slot, title, path):
        button = self.file_buttons[slot]
        button.setText(f"{title}\n{short_name(path)}")
        button.setToolTip(path)

    def _after_files(self):
        self._sync_left_checkbox()
        self._sync_frame_slider()
        if not self._range_ready:
            self.reset_frame_range()
        else:
            self._place_region()
        self.rebuild_images()
        self.redraw()

    def load_synthetic(self, ask=True, regenerate=False, root=None, n_frames=None):
        del ask  # The button loads the cubes directly. It does not open a dialog.
        # Windows cannot replace an .npy that is still memory-mapped.
        self._release_open_cubes()
        try:
            if regenerate:
                from bosoncompare.synthetic import generate
                cubes, logs = generate(root, n_frames=n_frames or N_FRAMES)
            else:
                cubes, logs = ensure(root, n_frames=n_frames or N_FRAMES)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Synthetic data", str(exc))
            self._after_files()
            return
        self._remember(str(cubes[0]))
        for index, path in enumerate(cubes):
            self._load_cube(index, os.fspath(path))
        loaded = self._loaded()
        if not loaded:
            return
        self._left_auto = True
        self.left_box.setChecked(True)
        self._range_ready = False
        self._after_files()
        names = ", ".join(os.path.basename(path) for path in logs)
        self.statusBar().showMessage(
            f"Synthetic cubes loaded ({loaded[0].nf} uint16 frames, heating "
            f"frames {HEAT_ON}–{HEAT_OFF}). Thermistor logs are next to them: {names}")

    def _release_open_cubes(self):
        for fs in self.sets:
            cube = fs.cube
            fs.cube = None
            fs.display_img = None
            if isinstance(cube, np.memmap):
                mapped = getattr(cube, "_mmap", None)
                if mapped is not None:
                    mapped.close()
            del cube
        gc.collect()

    def _fit_image_view(self, index):
        fs = self.sets[index]
        plot = self.img_plots[index]
        # Wide limits let the locked aspect pad instead of cropping the frame.
        plot.setLimits(xMin=-fs.nx, xMax=fs.nx * 2, yMin=-fs.ny, yMax=fs.ny * 2)
        plot.setRange(xRange=(0, fs.nx), yRange=(0, fs.ny), padding=0)

    # -------------------------------------------------------------- controls #
    def on_mode(self, mode):
        if mode == self.display_mode:
            return
        self.display_mode = mode
        if mode in ("tsr1", "tsr2") and not (self.frame_lo < self.frame <= self.frame_hi):
            mid = (self.frame_lo + self.frame_hi) // 2
            self.frame = max(mid, self.frame_lo + 1)
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(min(self.frame, self.frame_slider.maximum()))
            self.frame_slider.blockSignals(False)
            self.frame_label.setText(str(self.frame))
        self._move_frame_line()
        self.rebuild_images()
        self.redraw()

    def on_fps(self, value):
        if value <= 0:
            return
        self.fps = float(value)
        self._place_region()
        self.redraw()

    def on_frame_slider(self, value):
        self.frame = int(value)
        self.frame_label.setText(str(self.frame))
        self._move_frame_line()
        if self.display_mode in SLIDER_MODES:
            self.rebuild_images()
            self.draw_images()

    def on_t0(self, index, value):
        self.sets[index].t0 = float(value)
        if index == self._reference_index():
            self._place_region()
        self.redraw()

    def on_box(self, value):
        odd = int(value)
        if odd % 2 == 0:
            odd += 1
            self.box_spin.blockSignals(True)
            self.box_spin.setValue(odd)
            self.box_spin.blockSignals(False)
        self.half = odd // 2
        self.redraw_series()

    def on_ab(self, checked):
        self.show_ab = bool(checked)
        self.redraw_series()

    def on_subtract(self, checked):
        self.subtract_first = bool(checked)
        self.rebuild_images()
        self.redraw()

    def on_left_clicked(self, checked):
        self._left_auto = False
        if not checked and not any(fs.ready_log() for fs in self.sets):
            # No thermistor series exists, so the left axis stays on A − B.
            self._left_auto = True
            self.left_box.blockSignals(True)
            self.left_box.setChecked(True)
            self.left_box.blockSignals(False)
        self._set_axis_titles()
        self.redraw_series()

    def on_diff_pair(self, index):
        if index < 0:
            return
        self.diff_pair = DIFF_PAIRS[index]
        self.draw_images()

    def on_colormap(self, _index):
        self.draw_images()

    def _map_tip(self, difference):
        if difference:
            return ("Color scale for the difference view. "
                    "Auto uses the diverging CET-D1 map.")
        return ("Color scale for this cube. Auto uses inferno, "
                "or CET-D1 when the mode is signed. "
                "The number range stays shared across the three cubes.")

    def _resolved_map(self, combo, difference=False):
        chosen = combo.currentData()
        if chosen:
            return chosen
        if difference or self.display_mode in DIVERGING_MODES:
            return AUTO_DIVERGING
        return AUTO_SEQUENTIAL

    def _sync_left_checkbox(self):
        if not self._left_auto:
            return
        want = not any(fs.ready_log() for fs in self.sets)
        self.left_box.blockSignals(True)
        self.left_box.setChecked(want)
        self.left_box.blockSignals(False)

    def _difference_on_left(self):
        return left_axis_is_difference(
            self.left_box.isChecked(),
            any(fs.ready_log() for fs in self.sets))

    def _sync_frame_slider(self):
        nmax = 0
        for fs in self.sets:
            if fs.ready_cube():
                nmax = max(nmax, fs.nf - 1)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, max(nmax, 1))
        if self.frame > nmax:
            self.frame = nmax
        self.frame_slider.setValue(self.frame)
        self.frame_slider.blockSignals(False)
        self.frame_label.setText(str(self.frame))
        self._move_frame_line()

    # ---------------------------------------------------------- frame range #
    def _loaded(self):
        return [fs for fs in self.sets if fs.ready_cube()]

    def _reference_index(self):
        if self.sets[0].ready_cube():
            return 0
        for fs in self.sets:
            if fs.ready_cube():
                return fs.index
        return None

    def _frame_limit(self):
        loaded = self._loaded()
        if not loaded:
            return 1
        return max(fs.nf for fs in loaded)

    def _frame_to_t(self, frame):
        ref = self._reference_index()
        t0 = self.sets[ref].t0 if ref is not None else 0.0
        return t0 + float(frame) / self.fps

    def _t_to_frame(self, t):
        ref = self._reference_index()
        t0 = self.sets[ref].t0 if ref is not None else 0.0
        frame = int(round((float(t) - t0) * self.fps))
        return int(np.clip(frame, 0, self._frame_limit() - 1))

    def reset_frame_range(self):
        """Park the two lines on the first frame and the last frame."""
        last = self._frame_limit() - 1
        self.frame_lo, self.frame_hi = 0, max(last, 0)
        self._range_ready = bool(self._loaded())
        self._place_region()

    def _place_region(self):
        if not self._loaded():
            return
        last = self._frame_limit() - 1
        self.frame_lo = int(np.clip(self.frame_lo, 0, last))
        self.frame_hi = int(np.clip(self.frame_hi, 0, last))
        if self.frame_hi < self.frame_lo:
            self.frame_lo, self.frame_hi = self.frame_hi, self.frame_lo
        t0 = self._frame_to_t(0)
        t1 = self._frame_to_t(last)
        self._applying_region = True
        self.region.setBounds((t0, t1))
        self.region.setRegion((self._frame_to_t(self.frame_lo),
                               self._frame_to_t(self.frame_hi)))
        self._applying_region = False
        self._update_range_readout()
        self._move_frame_line()

    def _region_frames(self):
        lo_t, hi_t = self.region.getRegion()
        return self._t_to_frame(lo_t), self._t_to_frame(hi_t)

    def on_region_moved(self):
        if self._applying_region:
            return
        f0, f1 = self._region_frames()
        self._update_range_readout(f0, f1)

    def on_region_finished(self):
        if self._applying_region or not self._loaded():
            return
        f0, f1 = self._region_frames()
        if f0 == self.frame_lo and f1 == self.frame_hi:
            self._update_range_readout()
            return
        self.frame_lo, self.frame_hi = f0, f1
        self._range_ready = True
        self._update_range_readout()
        if self.display_mode in RANGE_MODES:
            self.rebuild_images()
            self.draw_images()
        self._publish_status(self._difference_on_left())

    def _update_range_readout(self, f0=None, f1=None):
        if f0 is None:
            f0, f1 = self.frame_lo, self.frame_hi
        count = abs(int(f1) - int(f0)) + 1 if self._loaded() else 0
        ref = self._reference_index()
        clock = SET_NAMES[ref] if ref is not None else "—"
        self.range_readout.setText(f"FRAME RANGE  {f0}  –  {f1}    {count} FRAMES    {clock} CLOCK")

    def _move_frame_line(self):
        self.frame_line.setPos(self._frame_to_t(self.frame))
        self.frame_line.setVisible(self.display_mode in SLIDER_MODES)

    def _window_for(self, fs):
        return (
            int(np.clip(self.frame_lo, 0, fs.nf - 1)),
            int(np.clip(self.frame_hi, 0, fs.nf - 1)),
        )

    # ---------------------------------------------------------------- images #
    def rebuild_images(self):
        for fs in self.sets:
            if not fs.ready_cube():
                fs.display_img = None
                continue
            f0, f1 = self._window_for(fs)
            fs.display_img = display_image(
                fs.cube, self.display_mode, frame=self.frame,
                subtract_first=self.subtract_first, f0=f0, f1=f1)

    def draw_images(self):
        shown = [fs.display_img for fs in self.sets if fs.display_img is not None]
        if self.display_mode in DIVERGING_MODES and shown:
            stacked = shown[0].ravel() if len(shown) == 1 else np.concatenate(
                [img.ravel() for img in shown])
            span = symmetric_limit(stacked)
            lo, hi = -span, span
        else:
            lo, hi = percentile_limits(shown)
        unit = mode_caption(self.display_mode, self.frame, self.frame_lo, self.frame_hi)
        used = []
        for index, fs in enumerate(self.sets):
            plot = self.img_plots[index]
            image = self.images[index]
            name = self._resolved_map(self.map_combos[index])
            image.setColorMap(name)
            if fs.display_img is None:
                image.clear()
                image.setVisible(False)
                plot.setTitle(f"{fs.name}  ·  no cube", color="#6d8290", size="10pt")
                continue
            used.append(name)
            image.setVisible(True)
            image.setImage(fs.display_img, autoLevels=False, levels=(lo, hi))
            image.setColorMap(name)
            plot.setTitle(
                f"<span style='color:{sample_color(index, 'A')}'>{fs.name}</span>"
                f"  {short_name(fs.cube_path)}"
                f"<br><span style='color:#8aa0b0'>{unit}</span>",
                color="#d7e4ec", size="10pt")
        if used and all(name == used[0] for name in used):
            bar_name = used[0]
            bar_label = unit
        else:
            bar_name = "CET-L1"
            bar_label = f"shared levels, {unit}"
        self.cbar.setColorMap(bar_name)
        self.cbar.setLevels((lo, hi))
        cbar_axis = self.cbar.getAxis("left")
        cbar_axis.enableAutoSIPrefix(False)
        cbar_axis.setLabel(bar_label, color="#8aa0b0")
        self._draw_marks()
        self._draw_diff(unit)

    def _draw_diff(self, unit):
        a, b = self.diff_pair
        left, right = self.sets[a], self.sets[b]
        title = f"{left.name} − {right.name}"
        if left.display_img is None or right.display_img is None:
            self.diff_image.clear()
            self.diff_image.setVisible(False)
            self.diff_plot.setTitle(f"{title}<br><span style='color:#6d8290'>load both cubes</span>",
                                    color="#d7e4ec", size="10pt")
            self.cbar_diff.setLevels((-1, 1))
            return
        if left.display_img.shape != right.display_img.shape:
            self.diff_image.clear()
            self.diff_image.setVisible(False)
            self.diff_plot.setTitle(
                f"{title}<br><span style='color:#6d8290'>shapes differ</span>",
                color="#d7e4ec", size="10pt")
            return
        diff = left.display_img - right.display_img
        span = symmetric_limit(diff)
        diff_name = self._resolved_map(self.diff_map_combo, difference=True)
        self.diff_image.setVisible(True)
        self.diff_image.setImage(diff, autoLevels=False, levels=(-span, span))
        self.diff_image.setColorMap(diff_name)
        self.cbar_diff.setColorMap(diff_name)
        self.cbar_diff.setLevels((-span, span))
        diff_axis = self.cbar_diff.getAxis("left")
        diff_axis.enableAutoSIPrefix(False)
        diff_axis.setLabel(title, color="#8aa0b0")
        ny, nx = diff.shape
        self.diff_plot.setLimits(xMin=-nx, xMax=nx * 2, yMin=-ny, yMax=ny * 2)
        self.diff_plot.setRange(xRange=(0, nx), yRange=(0, ny), padding=0)
        self.diff_plot.setTitle(
            f"{title}<br><span style='color:#8aa0b0'>warm = larger in {left.name}</span>",
            color="#d7e4ec", size="10pt")

    def _draw_marks(self):
        for mark in self._marks:
            view = mark.getViewBox()
            if view is not None:
                view.removeItem(mark)
        self._marks = []
        for loc, pick in enumerate(self.picks):
            if pick is None:
                continue
            px, py = pick
            for fs in self.sets:
                if not fs.ready_cube():
                    continue
                if not (0 <= px < fs.nx and 0 <= py < fs.ny):
                    continue
                mark = pg.ScatterPlotItem(
                    [px + 0.5], [py + 0.5], symbol=LOC_SYMBOLS[loc], size=16,
                    pen=pg.mkPen(sample_color(fs.index, LOC_NAMES[loc]), width=2),
                    brush=None)
                self.img_plots[fs.index].addItem(mark)
                self._marks.append(mark)

    # ---------------------------------------------------------------- series #
    def redraw(self):
        self._set_axis_titles()
        self.draw_images()
        self.redraw_series()

    def redraw_series(self):
        self._sync_right_view()
        self.legend.clear()
        self.series.clearPlots()
        self.right_vb.clear()
        # clearPlots removes every data item, including the range handles.
        if self.region.scene() is None:
            self.series.addItem(self.region)
        if self.frame_line.scene() is None:
            self.series.addItem(self.frame_line)

        left_values = []
        right_values = []
        diff_left = self._difference_on_left()

        if not diff_left:
            for fs in self.sets:
                if not fs.ready_log():
                    continue
                color = THERM_COLORS[fs.index]
                tag = short_name(fs.csv_path)
                t1 = self.series.plot(
                    fs.t_log, fs.t1, pen=pg.mkPen(color, width=1.6),
                    name=f"{fs.name} T1  {tag}")
                t2 = self.series.plot(
                    fs.t_log, fs.t2, pen=pg.mkPen(color, width=1.3, style=Qt.PenStyle.DashLine),
                    name=f"{fs.name} T2  {tag}")
                del t1, t2
                left_values.append(fs.t1)
                left_values.append(fs.t2)

        for loc, pick in enumerate(self.picks):
            if pick is None:
                continue
            px, py = pick
            for fs in self.sets:
                if not fs.ready_cube():
                    continue
                if not (0 <= px < fs.nx and 0 <= py < fs.ny):
                    continue
                series = extract_trace(
                    fs.cube, px, py, self.half, self.subtract_first, self.smooth_n, TLINEAR)
                label = f"{fs.name} Tf {LOC_NAMES[loc]}  x={px} y={py}"
                curve = pg.PlotCurveItem(
                    fs.t_cam(self.fps), series,
                    pen=pg.mkPen(sample_color(fs.index, LOC_NAMES[loc]), width=1.8))
                self.right_vb.addItem(curve)
                self.legend.addItem(curve, label)
                right_values.append(series)

        # The checkbox owns these traces. The left axis only decides where
        # they sit when there is no thermistor series to scale against.
        if self.show_ab and self._both_picks():
            for fs in self.sets:
                diff = self._difference_series(fs)
                if diff is None:
                    continue
                pen = pg.mkPen(sample_color(fs.index, "AB"), width=2.2)
                label = f"{fs.name} A−B"
                if diff_left:
                    self.series.plot(fs.t_cam(self.fps), diff, pen=pen, name=label)
                    left_values.append(diff)
                else:
                    curve = pg.PlotCurveItem(fs.t_cam(self.fps), diff, pen=pen)
                    self.right_vb.addItem(curve)
                    self.legend.addItem(curve, label)
                    right_values.append(diff)

        self._scale_axis(self.series.vb, left_values, empty=(0.0, 1.0))
        self._scale_axis(self.right_vb, right_values, empty=(0.0, 1.0))
        self._scale_time()
        self._publish_status(diff_left)

    def _both_picks(self):
        return self.picks[0] is not None and self.picks[1] is not None

    def _difference_series(self, fs):
        if not fs.ready_cube() or not self._both_picks():
            return None
        (ax, ay), (bx, by) = self.picks
        if not (0 <= ax < fs.nx and 0 <= ay < fs.ny and 0 <= bx < fs.nx and 0 <= by < fs.ny):
            return None
        series_a = extract_trace(
            fs.cube, ax, ay, self.half, self.subtract_first, self.smooth_n, TLINEAR)
        series_b = extract_trace(
            fs.cube, bx, by, self.half, self.subtract_first, self.smooth_n, TLINEAR)
        return series_a - series_b

    def _scale_axis(self, view, series_list, empty):
        finite = []
        for series in series_list:
            values = np.asarray(series, dtype=float)
            values = values[np.isfinite(values)]
            if values.size:
                finite.append(values)
        if not finite:
            view.setYRange(*empty, padding=0)
            return
        joined = finite[0] if len(finite) == 1 else np.concatenate(finite)
        lo, hi = pad_limits(float(joined.min()), float(joined.max()))
        view.setYRange(lo, hi, padding=0)

    def _scale_time(self):
        spans = []
        for fs in self.sets:
            if fs.ready_log():
                spans.append((float(fs.t_log[0]), float(fs.t_log[-1])))
            if fs.ready_cube():
                spans.append((fs.t0, fs.t0 + (fs.nf - 1) / self.fps))
        if not spans:
            self.series.setXRange(0.0, 1.0, padding=0)
            return
        lo = min(a for a, _ in spans)
        hi = max(b for _, b in spans)
        pad = 0.02 * max(hi - lo, 1e-9)
        self.series.setXRange(lo - pad, hi + pad, padding=0)

    def _publish_status(self, diff_left):
        parts = []
        for loc, pick in enumerate(self.picks):
            if pick is None:
                parts.append(f"{LOC_NAMES[loc]} none")
            else:
                parts.append(f"{LOC_NAMES[loc]} x={pick[0]} y={pick[1]}")
        parts.append(f"{self.fps:g} frames/s")
        for fs in self.sets:
            parts.append(f"{fs.name} t0={fs.t0:+.2f} s")
        if diff_left:
            if self._both_picks():
                parts.append("left axis scaled to A−B")
            else:
                parts.append("pick A and B to scale the left axis")
        else:
            parts.append("left axis thermistors")
        parts.append(f"frames {self.frame_lo}–{self.frame_hi}")
        self.statusBar().showMessage("    ".join(parts))

    def _sync_right_view(self):
        self.right_vb.setGeometry(self.series.vb.sceneBoundingRect())
        self.right_vb.linkedViewChanged(self.series.vb, self.right_vb.XAxis)

    # ----------------------------------------------------------------- input #
    def on_image_click(self, plot, ev):
        if ev.button() not in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            return
        if not plot.sceneBoundingRect().contains(ev.scenePos()):
            return
        if ev.button() == Qt.MouseButton.RightButton:
            self.clear_picks()
            ev.accept()
            return
        index = self.img_plots.index(plot)
        view = plot.vb.mapSceneToView(ev.scenePos())
        px = int(np.floor(view.x()))
        py = int(np.floor(view.y()))
        fs = self.sets[index]
        if not fs.ready_cube():
            return
        if not (0 <= px < fs.nx and 0 <= py < fs.ny):
            return
        self._take_pick((px, py))
        ev.accept()

    def _take_pick(self, pix):
        if self.picks[0] is None:
            self.picks[0] = pix
        else:
            self.picks[1] = pix
        self._draw_marks()
        self.redraw_series()

    def clear_picks(self):
        self.picks = [None, None]
        self._draw_marks()
        self.redraw_series()

    def nudge(self, index, delta):
        box = self.t0_boxes[index]
        box.setValue(box.value() + float(delta))

    def step_frame(self, delta):
        self.frame_slider.setValue(self.frame + int(delta))

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Type.KeyPress and self.isActiveWindow():
            focus = QApplication.focusWidget()
            if not isinstance(focus, (QAbstractSpinBox, QComboBox)):
                if self._handle_key(ev):
                    return True
        return super().eventFilter(obj, ev)

    def _handle_key(self, ev):
        text = ev.text()
        if text in ("c", "C"):
            self.clear_picks()
            return True
        if text == ",":
            self.nudge(0, -NUDGE)
            return True
        if text == ".":
            self.nudge(0, NUDGE)
            return True
        if text == "<":
            self.nudge(1, -10 * NUDGE)
            return True
        if text == ">":
            self.nudge(1, 10 * NUDGE)
            return True
        if text == ";":
            self.nudge(2, -NUDGE)
            return True
        if text == "'":
            self.nudge(2, NUDGE)
            return True
        if text == "[":
            self.step_frame(-1)
            return True
        if text == "]":
            self.step_frame(1)
            return True
        return False

    def closeEvent(self, ev):
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        super().closeEvent(ev)


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    font = QFont("Bahnschrift", 10)
    if font.exactMatch() or font.family() == "Bahnschrift":
        app.setFont(font)
    window = MainWindow()
    window.show()
    return app.exec()
