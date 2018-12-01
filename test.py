#!/usr/bin/env python3
from gaussian import gaussian

def test_x1_mu1_std1():
  actual = round(gaussian(1, 1, 1), 2)
  expected = 0.16
  assert actual == expected

def test_x1_mu05_std05():
  actual = round(gaussian(1, 0.5, 0.5), 2)
  expected = 0.39
  assert actual == expected