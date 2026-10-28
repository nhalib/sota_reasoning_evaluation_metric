"""Dataset items for Probe 1 (Uncertainty Coherence).
Contains verifiable arithmetic and logical reasoning items with mutually exclusive hypothesis pairs (H, not H)
and equivalent compiled ProbLog representations.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class Probe1Item:
    item_id: str
    premises: str
    hypothesis_h: str
    hypothesis_not_h: str
    ground_truth_h: int  # 1 if H is true, 0 if False
    problog_program: str


PROBE1_DATASET: List[Probe1Item] = [
    Probe1Item(
        item_id="math_01_apples",
        premises="Alice has 14 red apples. Bob gives Alice 8 green apples. Charlie takes 5 apples from Alice.",
        hypothesis_h="Alice currently has exactly 17 apples.",
        hypothesis_not_h="Alice currently has an apple count different from 17.",
        ground_truth_h=1,
        problog_program=r"""
        1.0::initial_apples(14).
        1.0::added_apples(8).
        1.0::removed_apples(5).
        total(T) :- initial_apples(A), added_apples(B), removed_apples(C), T is A + B - C.
        h :- total(17).
        not_h :- \+ h.
        query(h).
        query(not_h).
        """
    ),
    Probe1Item(
        item_id="math_02_perimeter",
        premises="A rectangular garden has length 12 meters and width 7 meters.",
        hypothesis_h="The perimeter of the garden is exactly 38 meters.",
        hypothesis_not_h="The perimeter of the garden is not 38 meters.",
        ground_truth_h=1,
        problog_program=r"""
        1.0::len(12).
        1.0::wid(7).
        perimeter(P) :- len(L), wid(W), P is 2 * (L + W).
        h :- perimeter(38).
        not_h :- \+ h.
        query(h).
        query(not_h).
        """
    ),
    Probe1Item(
        item_id="logic_03_transit",
        premises="Train A travels at 60 mph. Train B travels at 90 mph in the opposite direction. They start 300 miles apart.",
        hypothesis_h="The trains will meet in exactly 2.0 hours.",
        hypothesis_not_h="The trains will not meet in 2.0 hours.",
        ground_truth_h=1,
        problog_program=r"""
        1.0::speed_a(60).
        1.0::speed_b(90).
        1.0::dist(300).
        time_to_meet(T) :- speed_a(Sa), speed_b(Sb), dist(D), T is D / (Sa + Sb).
        h :- time_to_meet(2.0).
        not_h :- \+ h.
        query(h).
        query(not_h).
        """
    ),
    Probe1Item(
        item_id="prob_04_coin_urn",
        premises="A fair coin is tossed. If Heads, ball drawn from Urn 1 (3 red, 1 blue). If Tails, ball drawn from Urn 2 (1 red, 3 blue).",
        hypothesis_h="The probability that the drawn ball is red is exactly 0.50.",
        hypothesis_not_h="The probability that the drawn ball is red is not 0.50.",
        ground_truth_h=1,
        problog_program=r"""
        0.5::heads; 0.5::tails.
        0.75::red_u1.
        0.25::red_u2.
        red :- heads, red_u1.
        red :- tails, red_u2.
        h :- red.
        not_h :- \+ red.
        query(h).
        query(not_h).
        """
    ),
    Probe1Item(
        item_id="arithmetic_05_multiplication",
        premises="A warehouse contains 24 crates. Each crate holds 15 boxes. Each box contains 6 widgets.",
        hypothesis_h="The warehouse contains a total of 2,160 widgets.",
        hypothesis_not_h="The warehouse contains a widget total different from 2,160.",
        ground_truth_h=1,
        problog_program=r"""
        1.0::crates(24).
        1.0::boxes_per_crate(15).
        1.0::widgets_per_box(6).
        total_widgets(W) :- crates(C), boxes_per_crate(B), widgets_per_box(X), W is C * B * X.
        h :- total_widgets(2160).
        not_h :- \+ h.
        query(h).
        query(not_h).
        """
    ),
    Probe1Item(
        item_id="math_06_prime",
        premises="Consider the integer n = 91.",
        hypothesis_h="The integer n = 91 is a prime number.",
        hypothesis_not_h="The integer n = 91 is a composite number.",
        ground_truth_h=0,  # False! 91 = 7 * 13
        problog_program=r"""
        1.0::composite(91) :- true.
        h :- \+ composite(91).
        not_h :- composite(91).
        query(h).
        query(not_h).
        """
    ),
    Probe1Item(
        item_id="prob_07_medical_test",
        premises="Disease prevalence is 0.01. Test sensitivity is 0.90. False positive rate is 0.05. Patient tests positive.",
        hypothesis_h="The posterior probability that the patient has the disease is greater than 0.50.",
        hypothesis_not_h="The posterior probability that the patient has the disease is less than 0.50.",
        ground_truth_h=0,  # False! P(D|+) ~ 0.154 < 0.50
        problog_program=r"""
        0.01::disease.
        0.90::pos_given_d.
        0.05::pos_given_no_d.
        test_pos :- disease, pos_given_d.
        test_pos :- \+ disease, pos_given_no_d.
        evidence(test_pos, true).
        h :- disease.
        not_h :- \+ disease.
        query(h).
        query(not_h).
        """
    ),
    Probe1Item(
        item_id="math_08_discount",
        premises="A jacket originally costs $120. It is discounted by 25%, then an additional 10% coupon is applied to the sale price.",
        hypothesis_h="The final price of the jacket is exactly $81.00.",
        hypothesis_not_h="The final price of the jacket is different from $81.00.",
        ground_truth_h=1,  # 120 * 0.75 * 0.9 = 81
        problog_program=r"""
        1.0::orig(120.0).
        final(P) :- orig(O), P is O * 0.75 * 0.90.
        h :- final(81.0).
        not_h :- \+ h.
        query(h).
        query(not_h).
        """
    )
]
