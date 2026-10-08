#!/usr/bin/env python3
"""Boson Compare.

Compare three acquisitions. Each acquisition is a thermal cube (.npy, axis
order y, x, frame) plus an optional thermistor log (.csv with t_rel_s,
therm_c, therm2_c).

The window is a PySide6 console. Left-click a cube to sample that pixel in
every loaded cube: the first click is pixel A, later clicks move pixel B.
Right-click, or C, clears both. The camera traces sit on the right axis.
Thermistor traces sit on the left axis when a log is loaded. With no log,
the left axis is the difference between A and B, scaled to that difference.
The "Left axis: A − B" box forces that scale even when a log is open.

Two vertical lines on the time plot start at the first frame and the last
frame. The green line is the low frame and the orange line is the high
frame. Drag either line. Peak-to-peak, standard deviation, variance, mean,
principal components 1, 2, and 3, and the logarithmic derivatives use that
range. Each cube view and the difference view has its own color-scale menu.
Δ FRAME is the orange frame minus the green frame. TSR1 and TSR2 show the
Thermal Wave Imaging derivatives at the frame slider. The lines are drawn
on the S1 clock when S1 is loaded.
The same indexes are applied to every cube. The A − B checkbox adds those
difference traces, on the left axis when no thermistor log is loaded.

  left-click     set pixel A, then pixel B
  right-click    clear both pixels
  C              clear both pixels
  , / .          shift S1 camera time by -/+ 0.1 s
  < / >          shift S2 camera time by -/+ 1.0 s
  ; / '          shift S3 camera time by -/+ 0.1 s
  [ / ]          step the selected frame

The window opens with no cubes loaded. SYNTHETIC writes three uint16 cubes
of 1000 frames and loads them. Heating is off through frame 3, ramps from
frame 4, and the source shuts off at frame 240. One loaded cube is enough
for every frame view. The difference view waits until both cubes it names
are loaded.
"""

from bosoncompare.ui import main


if __name__ == "__main__":
    raise SystemExit(main())
