+++
title = "Release 0.7.0"
date = "2023-10-15"
+++

Linfa's 0.7.0 release mainly consists of improvements to Serde support. It also removes Approximate DBSCAN from `linfa-clustering` due to subpar performance and outdated dependencies.

<!-- more -->

## Improvements and fixes

 * Add `array_from_gz_csv` and `array_from_csv` in `linfa-datasets`.
 * Make Serde support in `linfa-linear`, `linfa-logistic`, and `linfa-ftrl` optional.
 * Add Serde support to `linfa-preprocessing` and `linfa-bayes`.
 * Bump `argmin` to 0.8.1.
 * Make licenses follow SPDX 2.1 license expression standard.

## Removals

Approximate DBSCAN is an alternative implementation of the DBSCAN algorithm that trades precision for speed. However, the implementation in `linfa-clustering` is actually slower than the regular DBSCAN implementation. It also depends on the `partitions` crate, which is incompatible with current versions of Rust. Thus, we have decided to remove Approximate DBSCAN from Linfa. The Approximate DBSCAN types and APIs are now aliases to regular DBSCAN.
