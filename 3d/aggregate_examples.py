examples = """

Example 1
Input:
Max height: 2
Grid size: 2 x 2
Top view:
1 0
0 1

Brief logic: Assign each '1' a valid height in [1,2]; ensure support rule holds.
Final output:
Layer 1:
1 0
0 1
Layer 2:
1 0
0 0

---

Example 2
Input:
Max height: 3
Grid size: 3 x 3
Top view:
1 0 1
0 1 0
1 0 0

Brief logic: Choose heights in [1,3]; construct layers bottom→top where a 1 in layer n means height ≥ n.
Final output:
Layer 1:
1 0 1
0 1 0
1 0 0
Layer 2:
1 0 0
0 0 0
1 0 0
Layer 3:
0 0 0
0 0 0
1 0 0
"""