---
date: '2025-04-14'
layout: single
tags:
- causal inference
- time series analysis
- personalized medicine
- digital health
- online experimentation
title: Core Problem
---

## Core Problem

N-of-1 trials (where a single subject serves as both treatment and control over time) traditionally require long "washout" periods between treatments to prevent interference effects. This paper addresses how to perform valid statistical inference when treatment effects persist over time, enabling more frequent treatment switching and shorter trials.


<!-- excerpt-end -->

## Key Framework

The authors combine causal inference with dynamic systems theory by:
- Modeling outcomes as a linear time-invariant system where current outcomes depend on past treatments
- Representing treatment effects as "impulse response functions" that show how effects decay over time
- Developing a generalized estimator that extends the classic Horvitz-Thompson approach to this setting

## Major Contributions

1. **Statistical Validity**: Proves asymptotic normality of the estimator, allowing for valid confidence intervals and hypothesis tests in N-of-1 trials with temporal interference

2. **Flexible Effect Measurement**: Introduces a framework to measure different aspects of treatment effects (immediate, cumulative, or comparative) using linear functionals

3. **No Washout Requirement**: Enables trials with frequent treatment switching rather than requiring long washout periods between treatments

4. **Theoretical Bridge**: Connects causal inference methods with system identification techniques from control theory

## Practical Applications

1. **Personalized Medicine**: More efficient testing of treatments for chronic conditions like pain management, psychiatric disorders, or diabetes

2. **Digital Health**: Evaluating mobile health interventions or behavioral nudges where effects persist over time

3. **Online Experimentation**: Testing algorithms, interfaces, or recommendations (as done by Netflix and LinkedIn) without waiting for effects to disappear

4. **Improved Patient Experience**: Shorter trials with frequent treatment switching can provide better average outcomes during the experimental period itself

## Limitations & Future Directions

- Assumes linear dynamics, which may not fully capture complex biological responses
- Mathematical complexity may limit immediate adoption without simplified guidelines
- Future work could extend the framework to nonlinear systems and time-varying effects

This paper makes a significant theoretical advance by showing that valid inference in N-of-1 trials is possible even with persistent treatment effects, potentially expanding the range of conditions where personalized medicine approaches can be rigorously evaluated.