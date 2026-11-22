"""Measure where a particular dog's identity lives, from phone footage.

tracks     keypoint tracks: load DeepLabCut output, save and load .npz
normalize  fill gaps, smooth, face right, remove size and travel; optional canonical skeleton
dots       render movement-only (point-light) clips
audio      extract sound, find claps and whistles
react      head direction and head-turn latency after a sound
gait       footfalls, stride timing, duty factor, diagonal dissociation
"""
