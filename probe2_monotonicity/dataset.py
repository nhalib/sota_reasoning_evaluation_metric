"""Dataset items for Probe 2 (Monotonicity & Defeasible Stability).
Operationalizes Definition 3.1: Monotonicity-Preserving Additions (Gamma |= phi).
Each item contains base premises Gamma, expected conclusion C, and three classes of entailed additions:
1. Conjunct expansion
2. Universal instantiation
3. Semantic paraphrase
plus equivalent Clingo (ASP) representations.
"""

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class Probe2Item:
    item_id: str
    base_premises: str
    question: str
    expected_conclusion: str
    additions: Dict[str, str]  # 'conjunct', 'instantiation', 'paraphrase'
    clingo_base: str
    clingo_additions: Dict[str, str]


PROBE2_DATASET: List[Probe2Item] = [
    Probe2Item(
        item_id="mono_01_animals",
        base_premises=(
            "All raptors are birds. All eagles are raptors.\n"
            "Aquila is an eagle and Aquila is healthy."
        ),
        question="Is Aquila a bird? Answer with 'Yes' or 'No'.",
        expected_conclusion="Yes",
        additions={
            "conjunct": "Additionally, we know that Aquila is an eagle.",
            "instantiation": "Additionally, we know that Aquila is a raptor.",
            "paraphrase": "Additionally, note that every raptor belongs to the bird family."
        },
        clingo_base="""
        bird(X) :- raptor(X).
        raptor(X) :- eagle(X).
        eagle(aquila).
        healthy(aquila).
        query_target :- bird(aquila).
        #show query_target/0.
        """,
        clingo_additions={
            "conjunct": "eagle(aquila).",
            "instantiation": "raptor(aquila).",
            "paraphrase": "bird(X) :- raptor(X)."
        }
    ),
    Probe2Item(
        item_id="mono_02_transit",
        base_premises=(
            "If train A departs at 08:00 and track 1 is clear, train A arrives at 09:30.\n"
            "Train A departs at 08:00 and track 1 is clear."
        ),
        question="Does train A arrive at 09:30? Answer with 'Yes' or 'No'.",
        expected_conclusion="Yes",
        additions={
            "conjunct": "Additionally, we know that track 1 is clear.",
            "instantiation": "Additionally, note that train A departs at 08:00.",
            "paraphrase": "Additionally, note that whenever track 1 is unobstructed, departure at 08:00 guarantees arrival at 09:30."
        },
        clingo_base="""
        departs(trainA, 800).
        clear(track1).
        arrives(trainA, 930) :- departs(trainA, 800), clear(track1).
        query_target :- arrives(trainA, 930).
        #show query_target/0.
        """,
        clingo_additions={
            "conjunct": "clear(track1).",
            "instantiation": "departs(trainA, 800).",
            "paraphrase": "arrives(trainA, 930) :- departs(trainA, 800), clear(track1)."
        }
    ),
    Probe2Item(
        item_id="mono_03_inventory",
        base_premises=(
            "Every item in bin 4 is a steel bolt.\n"
            "Part 108 is in bin 4 and part 108 is zinc-plated."
        ),
        question="Is part 108 a steel bolt? Answer with 'Yes' or 'No'.",
        expected_conclusion="Yes",
        additions={
            "conjunct": "Additionally, we know that part 108 is in bin 4.",
            "instantiation": "Additionally, we know that part 108 is a steel bolt.",
            "paraphrase": "Additionally, note that any component stored inside bin 4 is classified as a steel bolt."
        },
        clingo_base="""
        steel_bolt(X) :- in_bin4(X).
        in_bin4(part108).
        zinc_plated(part108).
        query_target :- steel_bolt(part108).
        #show query_target/0.
        """,
        clingo_additions={
            "conjunct": "in_bin4(part108).",
            "instantiation": "steel_bolt(part108).",
            "paraphrase": "steel_bolt(X) :- in_bin4(X)."
        }
    ),
    Probe2Item(
        item_id="mono_04_discount",
        base_premises=(
            "All premium members receive a 20% discount on electronics.\n"
            "David is a premium member and David purchases a laptop."
        ),
        question="Does David receive a 20% discount on his laptop? Answer with 'Yes' or 'No'.",
        expected_conclusion="Yes",
        additions={
            "conjunct": "Additionally, we know that David is a premium member.",
            "instantiation": "Additionally, we know that David receives a 20% discount on electronics.",
            "paraphrase": "Additionally, note that holding premium membership entitles the buyer to 20% off all electronic equipment."
        },
        clingo_base="""
        electronics(laptop).
        premium_member(david).
        purchases(david, laptop).
        discount20(User, Item) :- premium_member(User), purchases(User, Item), electronics(Item).
        query_target :- discount20(david, laptop).
        #show query_target/0.
        """,
        clingo_additions={
            "conjunct": "premium_member(david).",
            "instantiation": "purchases(david, laptop).",
            "paraphrase": "discount20(User, Item) :- premium_member(User), purchases(User, Item), electronics(Item)."
        }
    ),
    Probe2Item(
        item_id="mono_05_prerequisites",
        base_premises=(
            "Passing Math 101 is required to enroll in Math 201.\n"
            "Passing Math 201 is required to enroll in Math 301.\n"
            "Elena has enrolled in Math 301 and Elena is an honors student."
        ),
        question="Has Elena passed Math 101? Answer with 'Yes' or 'No'.",
        expected_conclusion="Yes",
        additions={
            "conjunct": "Additionally, we know that Elena is an honors student.",
            "instantiation": "Additionally, we know that Elena has passed Math 201.",
            "paraphrase": "Additionally, note that enrollment in Math 201 strictly necessitates having previously passed Math 101."
        },
        clingo_base="""
        enrolled(elena, math301).
        honors(elena).
        passed(X, math201) :- enrolled(X, math301).
        passed(X, math101) :- passed(X, math201).
        query_target :- passed(elena, math101).
        #show query_target/0.
        """,
        clingo_additions={
            "conjunct": "honors(elena).",
            "instantiation": "passed(elena, math201).",
            "paraphrase": "passed(X, math101) :- passed(X, math201)."
        }
    ),
    Probe2Item(
        item_id="mono_06_chemistry",
        base_premises=(
            "All noble gases have full outer electron shells.\n"
            "Argon is a noble gas and argon is in period 3."
        ),
        question="Does Argon have a full outer electron shell? Answer with 'Yes' or 'No'.",
        expected_conclusion="Yes",
        additions={
            "conjunct": "Additionally, we know that argon is in period 3.",
            "instantiation": "Additionally, we know that argon has a full outer electron shell.",
            "paraphrase": "Additionally, note that possessing a full outer electron shell is a property of every noble gas."
        },
        clingo_base="""
        noble_gas(argon).
        period3(argon).
        full_shell(X) :- noble_gas(X).
        query_target :- full_shell(argon).
        #show query_target/0.
        """,
        clingo_additions={
            "conjunct": "period3(argon).",
            "instantiation": "full_shell(argon).",
            "paraphrase": "full_shell(X) :- noble_gas(X)."
        }
    )
]
