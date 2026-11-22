Give realistic values for adult dogs of two breeds, the Labrador Retriever and the Belgian Malinois, for building a procedural animation of a trotting dog seen from the side. Use the veterinary gait-analysis and canine biomechanics literature where you can, cite it (first author and year), and mark anything that is your estimate rather than a measured value. Be honest about uncertainty; ranges are better than false precision.

Return ONLY a JSON array, no prose before or after. Each element:
{"param": "...", "unit": "...", "lab_typical": number, "lab_range": [lo, hi], "mal_typical": number, "mal_range": [lo, hi], "source": "author year, or 'estimate'", "confidence": "high|medium|low", "note": "short"}

Parameters:
1. withers_height (m)
2. body_length_to_height_ratio (point of shoulder to point of buttock, divided by withers height)
3. body_mass (kg)
4. preferred_trot_speed (m/s)
5. trot_stride_frequency (Hz, full strides per second)
6. trot_stride_length (m)
7. trot_duty_factor (fraction of the stride a foot is on the ground)
8. walk_duty_factor
9. walk_to_trot_transition_speed (m/s)
10. trot_diagonal_dissociation (hind foot touchdown before its diagonal fore foot, as a fraction of the stride; negative if the fore lands first)
11. trot_vertical_trunk_oscillation (peak-to-peak, m, at the withers)
12. trot_neck_angle_above_horizontal (degrees)
13. scapula_rotation_range_trot (degrees)
14. carpal_flexion_max_in_swing (degrees of flexion from straight)
15. tarsal_hock_angle_range_trot (min and max joint angle, degrees)
16. standing_joint_angles: shoulder, elbow, carpus, hip, stifle, hock (one element each, degrees)
17. segment lengths as a fraction of withers height: scapula, humerus, radius_ulna, metacarpus_plus_paw, femur, tibia, metatarsus_plus_paw (one element each)
18. resting_respiratory_rate (breaths per minute)
19. panting_rate (breaths per minute)
20. tail_carriage_at_trot (degrees above the line of the back)
21. tail_wag_frequency (Hz)
22. head_bob_amplitude_trot (degrees, sound dog)
23. foot_swing_height_fore_and_hind_trot (as a fraction of leg length; one element each)
