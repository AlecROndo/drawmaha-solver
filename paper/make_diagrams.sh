#!/bin/bash
# Graphviz diagrams for the drawmaha ML survey (house palette).
set -e
cd "$(dirname "$0")"
mkdir -p figures

INK="#1A202C"; SLATE="#52616F"; TEAL="#0F7B6C"; BLUE="#2B6CB0"
RED="#C53030"; TEALPALE="#EDF7F3"; PALEBLUE="#EEF2F7"; HAIR="#C9CFD6"; GRAYPALE="#F4F5F3"

# ---- 1. Drawmaha street structure -------------------------------------------
dot -Tpng -Gdpi=200 -o figures/streets.png <<EOF
digraph streets {
  rankdir=LR; bgcolor=white;
  node [fontname="Helvetica", fontsize=11, shape=box, style="rounded,filled",
        fillcolor="$PALEBLUE", color="$HAIR", fontcolor="$INK", margin="0.14,0.08"];
  edge [color="$SLATE", arrowsize=0.7, penwidth=1.0];
  deal   [label="Deal\n5 + 5 cards", fillcolor="$GRAYPALE"];
  bet1   [label="Betting\n(preflop)"];
  flop   [label="Flop\n3 board cards", fillcolor="$GRAYPALE"];
  bet2   [label="Betting"];
  draw   [label="DRAW\ndiscard 0-5, replace\n(count is public)", fillcolor="$TEALPALE", color="$TEAL", fontcolor="$TEAL"];
  turn   [label="Turn", fillcolor="$GRAYPALE"];
  bet3   [label="Betting"];
  river  [label="River", fillcolor="$GRAYPALE"];
  bet4   [label="Betting"];
  show   [label="Showdown\nhalf: 5-card hand\nhalf: Omaha hand", fillcolor="#FBEAE8", color="$RED", fontcolor="$RED"];
  deal -> bet1 -> flop -> bet2 -> draw -> turn -> bet3 -> river -> bet4 -> show;
}
EOF

# ---- 2. Deep CFR training pipeline ------------------------------------------
dot -Tpng -Gdpi=200 -o figures/pipeline.png <<EOF
digraph pipeline {
  rankdir=TB; bgcolor=white; ranksep=0.32; nodesep=0.25;
  node [fontname="Helvetica", fontsize=10.5, shape=box, style="rounded,filled",
        fillcolor="$PALEBLUE", color="$HAIR", fontcolor="$INK", margin="0.13,0.07"];
  edge [color="$SLATE", fontcolor="$SLATE", fontname="Helvetica", fontsize=9, arrowsize=0.7];

  traverse [label="External-sampling traversals of the real game\n(traverser tries every action; opponent & chance sample one)", fillcolor="$TEALPALE", color="$TEAL"];
  advbuf  [label="Advantage buffer (per player)\nreservoir-sampled (infoset, sampled regret, t)"];
  stratbuf [label="Strategy buffer\n(infoset, current strategy, t)"];
  advnet  [label="Advantage network (per player)\nretrained FROM SCRATCH each iteration", fillcolor="$TEALPALE", color="$TEAL"];
  rm      [label="Regret matching over predicted advantages\n= next iteration's strategy"];
  avgnet  [label="Average-policy network\ntrained ONCE at the end, samples weighted by t", fillcolor="$TEALPALE", color="$TEAL"];
  dash    [label="Dashboard / play\n(one forward pass per query)", fillcolor="#FBEAE8", color="$RED", fontcolor="$RED"];

  traverse -> advbuf [label=" sampled regrets"];
  traverse -> stratbuf [label=" visited strategies"];
  advbuf -> advnet [label=" regression"];
  advnet -> rm;
  rm -> traverse [label=" defines play for\n next traversals", style=dashed];
  stratbuf -> avgnet [label=" distillation"];
  avgnet -> dash;
}
EOF

# ---- 3. Lineage map: four families ------------------------------------------
dot -Tpng -Gdpi=200 -o figures/lineage.png <<EOF
digraph lineage {
  rankdir=TB; bgcolor=white; ranksep=0.28; nodesep=0.5; newrank=true;
  node [fontname="Helvetica", fontsize=10, shape=box, style="rounded,filled",
        fillcolor="$PALEBLUE", color="$HAIR", fontcolor="$INK", margin="0.11,0.05"];
  edge [color="$SLATE", arrowsize=0.65];

  subgraph cluster_tab {
    label="TABULAR CFR"; fontname="Helvetica-Bold"; fontsize=10.5; fontcolor="$BLUE"; color="$HAIR"; style=rounded;
    cfr    [label="CFR (2007)\nZinkevich et al."];
    mccfr  [label="MCCFR (2009)\nLanctot et al."];
    cfrp   [label="CFR+ (2014)\nTammelin"];
    ceph   [label="Cepheus (2015)\nlimit hold'em solved"];
    dcfr   [label="Discounted CFR (2019)\nBrown & Sandholm"];
    cfr -> mccfr; cfr -> cfrp; cfrp -> ceph; cfrp -> dcfr;
  }
  subgraph cluster_neural {
    label="NEURAL CFR"; fontname="Helvetica-Bold"; fontsize=10.5; fontcolor="$TEAL"; color="$HAIR"; style=rounded;
    deepcfr [label="Deep CFR (2019)\nBrown et al.", fillcolor="$TEALPALE", color="$TEAL"];
    sdcfr   [label="SD-CFR (2019)\nSteinberger"];
    dream   [label="DREAM (2020)\nmodel-free"];
    escher  [label="ESCHER (2022)\nno importance sampling"];
    deepcfr -> sdcfr; sdcfr -> dream; dream -> escher;
  }
  subgraph cluster_search {
    label="SEARCH + VALUE NETS"; fontname="Helvetica-Bold"; fontsize=10.5; fontcolor="$INK"; color="$HAIR"; style=rounded;
    decomp  [label="Safe re-solving (2014)\nBurch et al."];
    deepstack [label="DeepStack (2017)\ncontinual resolving"];
    libratus  [label="Libratus (2017)\nnested subgame solving"];
    dls     [label="Depth-limited (2018)\nModicum; Pluribus 2019"];
    rebel   [label="ReBeL (2020)\nRL + search on beliefs"];
    sog     [label="Student of Games (2023)"];
    gtow    [label="GTO Wizard AI (2023)\ncommercial", fillcolor="#FBEAE8", color="$RED"];
    decomp -> deepstack; decomp -> libratus; libratus -> dls; dls -> rebel; rebel -> sog; rebel -> gtow;
  }
  subgraph cluster_rl {
    label="RL EQUILIBRIUM"; fontname="Helvetica-Bold"; fontsize=10.5; fontcolor="$SLATE"; color="$HAIR"; style=rounded;
    nfsp  [label="NFSP (2016)\naverage the policies"];
    psro  [label="PSRO (2017)\npopulation meta-game"];
    rnad  [label="R-NaD / DeepNash (2022)\nregularized dynamics"];
    mmd   [label="MMD (2023)\nmirror descent + magnet"];
    nfsp -> psro; psro -> rnad; rnad -> mmd;
  }
  cfr -> deepcfr [style=dashed, constraint=false];
  cfr -> decomp  [style=dashed, constraint=false];
}
EOF

# ---- 4. The validation ladder ------------------------------------------------
dot -Tpng -Gdpi=200 -o figures/ladder.png <<EOF
digraph ladder {
  rankdir=TB; bgcolor=white; ranksep=0.3;
  node [fontname="Helvetica", fontsize=10.5, shape=box, style="rounded,filled",
        fillcolor="$PALEBLUE", color="$HAIR", fontcolor="$INK", margin="0.15,0.09"];
  edge [color="$SLATE", fontcolor="$SLATE", fontname="Helvetica", fontsize=9.5, arrowsize=0.7];

  rps  [label="Rock-paper-scissors -- regret matching\nproves: the ledger update      check: average -> 1/3 each"];
  kuhn [label="Kuhn poker -- vanilla CFR (tabular)\nproves: CFR on a tree      check: known closed-form equilibrium"];
  leduc [label="Leduc poker -- CFR+ / DCFR (tabular)\nproves: board cards, faster variants      check: OpenSpiel reference"];
  mini [label="Mini-drawmaha -- external-sampling MCCFR\nproves: split pot, draw round, face-up draw-1 rule\ncheck: exact best-response tree walk", fillcolor="$TEALPALE", color="$TEAL"];
  full [label="Full drawmaha -- Deep CFR\nproves: the headline result\ncheck: mini-game recipe repeated + best-response probe + head-to-head", fillcolor="$TEALPALE", color="$TEAL"];

  rps -> kuhn [label=" add: a game tree"];
  kuhn -> leduc [label=" add: board cards"];
  leduc -> mini [label=" add: drawmaha's rules"];
  mini -> full [label=" swap: table -> networks"];
}
EOF

echo "diagrams done"
ls -la figures/*.png
