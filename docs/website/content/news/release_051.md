+++
title = "Release 0.5.1"
date = "2022-02-28"
+++

Linfa's 0.5.1 release fixes errors and bugs in the previous release, as well as removing useless trait bounds on the `Dataset` type. Note that the commits for this release are located in the `0-5-1` branch of the GitHub repo.

## Improvements

 * remove `Float` trait bound from many `Dataset` impls, making non-float datasets usable
 * fix build errors in 0.5.0 caused by breaking minor releases from dependencies
 * fix bug in k-means where the termination condition of the algorithm was calculated incorrectly
 * fix build failure when building `linfa` alone, caused by incorrect feature selection for `ndarray`
