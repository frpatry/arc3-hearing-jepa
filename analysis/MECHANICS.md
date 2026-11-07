# Mechanics of the 25 official ARC-AGI-3 public games

Purpose: understand how the official games work so that the v4 procedural generator (`gen/v4/`) can reproduce the *family* (kinds of rules, interfaces, escalation, visual statistics) with its own designs.
No official code, level layouts or sprites are copied here or in the generator; the game files were read and run offline only to understand their mechanics.
Sources: four study notes (games run in the offline engine, every level rendered, 300 random actions per level) plus `REPORT.md` (short fiches and generated-vs-official statistics).

## Summary table

"Random no-effect" is the fraction of random actions with no visible world change on the first three levels (as reported by the notes; usually "nothing changed outside the step bar"; see the per-game entry for the exact measure).

| game | control | core mechanic | win | levels | random no-effect (first 3 levels) | background / style |
|---|---|---|---|---|---|---|
| ar25 | select + arrows (click / A5 select), undo | mirrors reflect pieces recursively | cover target silhouette | 8 | 0.29-0.49 | flat blue, 3-px cells |
| bp35 | avatar (left/right) + click terrain, undo | edit terrain under inverted gravity | reach the gem | 9 | 0.44 / 0.45 / 0.40 (bar-only) | black rock, light-blue air |
| cd82 | basket on ring + click palette + A5 pour | layered half-canvas painting | canvas = target thumbnail | 6 | 0.67 / 0.63 / 0.60 | black, top HUD band |
| cn04 | select + arrows + A5 rotate | connector-matching jigsaw | all connectors paired | 6 | 0.38 / 0.22 / 0.32 | saturated single colour |
| dc22 | avatar + click side panel | letter-keyed toggle groups, crane | avatar on goal | 6 | 0.38-0.51 | off-black, right panel |
| ft09 | click only | colour cycling under neighbour constraints | all constraint tiles satisfied | 6 | 0.99 / 0.63 / 0.61 | dark grey, 6-px tiles |
| g50t | avatar + A5 rewind | recorded ghosts hold pressure plates | player in dock | 7 | 0.48 / 0.40 / 0.61 | white, black corridors |
| ka59 | select (click) + arrows | kick boxes into size-matched frames | every frame holds its box | 7 | 0.39 / 0.27 / 0.20 | light-grey rooms, grey walls |
| lf52 | click select/jump + arrows (ferries), undo | peg solitaire with rail ferries | one peg (per colour) left | 10 | 0.98 / 0.54 / 0.40 (bar-only) | light blue, framed islands |
| lp85 | click only (arrow buttons) | rotating rings sharing tiles | marked tiles in markers | 8 | 0.95 / 0.93 / 0.98 | dark-grey board, tiny tiles |
| ls20 | avatar only | reshape carried key at stations | every door opened | 7 | 0.27-0.50 (world) | grey maze, bottom HUD |
| m0r0 | mirrored twin avatars + click blocks | desync mirrored twins via walls | twins merge | 6 | 0.53 / 0.55 / 0.51 | black floor, two-tone walls |
| r11l | click only (select + place) | body sits at legs' centroid | bodies on targets (+colour sets) | 6 | 0.86 / 0.62 / 0.52 | black field, grey blobs |
| re86 | A5 select + arrows | move/recolour/deform 1-px line drawings | coloured lines through all dots | 8 | 0-0.07 world (0.18-0.31 at ≤4 px) | black, thin bright lines |
| s5i5 | click only (colour panels) | stretch/rotate colour-keyed arm chains | bar tips on targets | 8 | 0.77 / 0.69 / 0.77 | black, bottom panels |
| sb26 | click (palette, slots) + A5 run, undo | colour program with subroutine calls | run emits target sequence | 8 | 0.90 / 0.88 / 0.86 | dark, palette strip |
| sc25 | avatar + click rune keypad | runes cast spells (resize, fire, teleport) | walk into exit | 6 | 0.27 / 0.29 / 0.22 | black, grey corridors |
| sk48 | arm arrows + click switch, undo | telescoping arm ferries coloured blocks | arm colours = reference sequence | 8 | 0.28 / 0.27 / 0.26 | black, dark-grey grid |
| sp80 | select + arrows + A5 spill | route falling liquid into cups | all cups filled, no hazard hit | 6 | 0.45 / 0.36 / 0.36 | orange, flat bars |
| su15 | click pulse + undo | suction pulse, Suika merges, chasers | goal zones hold recipe | 9 | 0.76-0.97 (uniform clicks) | black, top recipe HUD |
| tn36 | click only (bits, run, tabs) | binary opcodes drive a piece | piece pose = target | 7 | 0.81 / 0.57 / 0.41 (bar-only) | black, two machine panels |
| tr87 | keyboard cursor only | apply glyph rewrite rules | output = translated input | 6 | 0.00 | grey panels, dense glyphs |
| tu93 | avatar only | maze with turn-based enemies | stand on exit | 9 | 0.59 / 0.59 / 0.48 | black, checkered track |
| vc33 | click only | pumps shift communicating vessels | markers level with notches | 7 | 0.92-0.97 (object clicks) | grey, white fluid |
| wa30 | avatar + A5 grab | hauling with helper / thief NPCs | boxes in goals, unheld | 9 | 0.22 / 0.00 / 0.11 | very light grey, 4-px grid |

## Per-game entries

### ar25 — mirror kaleidoscope cover puzzle
- **Actions**: arrows move the selected object one cell (vertical mirrors only left/right, horizontal ones only up/down); ACTION5 cycles the selection; ACTION6 selects the piece or mirror under the cursor (alternates at a mirror crossing); ACTION7 undoes. Clicks and undo are free, moves and ACTION5 cost a step.
- **Objects & interactions**: black polyomino pieces, light-blue dotted mirror axes, yellow target silhouettes, locked objects. Each piece is reflected through every mirror recursively, so two perpendicular mirrors give 4-fold copies; dark-grey reflections count as coverage. Code for one-axis and rotating pieces exists but no level uses it.
- **Win**: every target cell covered by a piece or a reflection.
- **Escalation**: 8 levels on a fixed 21×21 board (baselines 32/50/75/37/89/159/233/73). L0 one piece + locked vertical mirror; L1 movable mirror; L2-3 horizontal mirror and 2 pieces; L4 both mirrors; L5-7 both mirrors, 2 pieces, 42-60 target cells.
- **No-effect**: world no-op 0.29-0.49 under random play (public 0.11). Empty or locked-object clicks do nothing (and are free); ACTION5 changes only selection markers; moves fail at the edge; undo with empty history.
- **Style**: flat blue background 72-89% empty, ≤6 colours, scale 3; status dots per cell; step bar in the right column, stacked colour layers when the budget exceeds 64.
- **Dynamics**: single-frame actions, except an 8-frame hint blink in L1 if the player never switches selection. Nothing autonomous.

### bp35 — upside-down gravity platformer with clickable terrain
- **Actions**: ACTION3/4 walk one 6-px cell; ACTION6 clicks a terrain cell; ACTION7 undo (costs a step). ACTION1, 2, 5 are not offered.
- **Objects & interactions**: gravity starts pointing up, so the player walks under the rock and falls (up or down) over gaps. Green blocks break when clicked, orange ones toggle solid/passable, red ones flip gravity and vanish, purple ones (L7+) spread. Spikes kill on landing; on L0 and L2 a rising tide advances every second move and kills.
- **Win**: reach the gem.
- **Escalation**: 9 levels (baselines 21→163): breakables + tide, then spikes, toggles, gravity flips, flip columns, spreading blocks and dense mixes. Levels are taller than the screen.
- **No-effect**: true no-op 0 (the bar always moves) but bar-only 0.44/0.45/0.40 on L0-2 and 0.38 on L5; most clicks (62/74, 88/90, 80/80) hit inert rock, air, spikes or the player; also wall bumps and empty undo. Public figure (0.27) is corrupted (see caveats).
- **Style**: dotted black rock, light-blue air, 6-px cells, solid coloured blocks, 4×5 player with a yellow eye; about half the layout is rock.
- **Dynamics**: moves ~5 frames, falls up to ~20 with an easing camera that follows vertically; 3-frame morphs for block changes; the tide moves on its own every second action. Budget 64 (128 from L7).

### cd82 — paint bucket orbiting a canvas
- **Actions**: arrows move a basket around 8 ring positions surrounding a 10×10 canvas (centre forbidden, outward clamped); ACTION5 pours; ACTION6 on a palette swatch picks a colour; from L3 ACTION6 on the nozzle fires an edge stamp (only from edge-centre positions).
- **Objects & interactions**: a side pour paints a rectangular half, a corner pour a triangular half, the stamp a small strip. Later paint covers earlier paint, so order matters.
- **Win**: canvas equals the target thumbnail (diagonal seams ignored).
- **Escalation**: 6 levels (baselines 55/8/41/21/23/23); palette 2→3→7 colours; straight halves, then diagonals, then stamps and 4-6-region targets.
- **No-effect**: 0.67/0.63/0.60 then 0.54-0.59. Clicks 80-100% inert (only palette/nozzle, re-picking the same colour does nothing), arrows ~50% blocked by clamping, pours 60-80% repaint the same colour.
- **Style**: black background ~79% empty; grey top HUD with framed target preview and swatch row with underline; white canvas; ~12×6 bucket.
- **Dynamics**: a pour is a 15-frame out-and-back animation; budget 100, no refill.

### cn04 — jigsaw with connectors
- **Actions**: ACTION6 selects a piece (it turns white), re-click deselects or cycles stacked alternative shapes; arrows move it one cell (3 px), clamped; ACTION5 rotates 90° or cycles variants. Without a selection, moves and rotation do nothing.
- **Objects & interactions**: 2-13 thick-line glyph pieces with red connector stubs; overlapping connector pixels turn dark grey. Pieces pass over each other. Hidden rule: two connector kinds both drawn red; only same-kind pairs count.
- **Win**: every connector paired with one of the same kind.
- **Escalation**: 6 levels on a 20×20 board (×3), background colour changes per level (baselines 29/54/85/300/208/113). L0 2 pieces, L1 4; from L2 unselected pieces are grey silhouettes (memory required); L4 8 pieces + a variant stack; L5 13 pre-rotated pieces.
- **No-effect**: 0.38/0.22/0.32 (public 0.06): actions with nothing selected, edge clamps, rotating symmetric pieces, empty clicks. Every action costs a step.
- **Style**: plain saturated background 85-90% of the screen; 1-cell strokes; white = selected; 32-px step bar centred on row 0.
- **Dynamics**: instant, one frame, nothing autonomous; connector feedback updates every action.

### dc22 — walk-and-toggle bridges with a side control panel
- **Actions**: arrows move a 2×2 avatar 2 px; ACTION6 clicks lettered hat buttons in a black right panel, each cycling every world object with that letter to its next variant; later, arrow and grab buttons drive a crane.
- **Objects & interactions**: floor pieces, blocks, letter-keyed toggle groups, carrier tiles that move the avatar, keys that reveal hidden buttons, plates that show buttons only while stood on, a crane that carries bridge segments, colour-cycling tiles (L5). If floor vanishes under the avatar it falls.
- **Win**: avatar on the goal tile.
- **Escalation**: 6 levels (baselines 59/102/67/98/324/578, budgets 128-1024). L0 2 groups + tutorial outline; L1 key; L2 carriers; L3 more letters and clutter; L4 crane, full 64×64; L5 everything (8 letters, ~100 sprites).
- **No-effect**: world no-op 0.38-0.51: moves blocked 25-50% (walls, off-floor), 73/76 object-pixel clicks inert, hidden buttons make even valid clickables fail ~50% in L1. Every click and blocked move costs; a press costs 2.
- **Style**: off-black, grey letterbox, scale 1, dashed divider to the panel; multicolour checkered sprites; 31-47% background.
- **Dynamics**: a fall is a 15-frame vignette costing 20 steps, then the click is undone; failed crane shake 3 frames, failed grab blink 6.

### ft09 — colour-cycling constraint puzzle
- **Actions**: ACTION6 only. A tile click advances its colour in the level palette (shown top right); from L4 patterned tiles advance every tile their pattern points to (Lights-Out style).
- **Objects & interactions**: plain, patterned and constraint tiles; a constraint is a 3×3 icon whose 8 outer pixels say "neighbour equals centre colour" (white) or "differs" (grey).
- **Win**: all constraints satisfied simultaneously.
- **Escalation**: 6 levels (baselines 43/12/23/28/65/37; budgets 32-128). L0 3×3 tutorial with three solved example boards; L1-2 5×5 two colours; L3 three colours; L4 patterns; L5 almost all tiles patterned.
- **No-effect**: 0.99/0.63/0.61; clicks on constraints, swatches, gaps and background are inert and free. Public 1.00 is a harvesting artefact (caveats).
- **Style**: dark grey, 32×32 grid at ×2 (6-px tiles, 2-px gaps), board ~40% of the screen, orange/yellow bottom bar.
- **Dynamics**: one frame per click; a 5-frame blink on L0 off-board clicks leaves the end frame unchanged.

### g50t — time-loop maze with ghosts
- **Actions**: arrows slide one 6-px cell; ACTION5 rewinds: the history plays backwards, the player returns to start and the recorded run becomes a ghost that replays in step with later moves. Three loop slots; the last clears all ghosts.
- **Objects & interactions**: colour-wired plates hold gates and moving bars while the player or a ghost stands on them; purple multi-plate links; from L5 green patrol bots that step per move and kill on contact.
- **Win**: the player (not a ghost) in the dock.
- **Escalation**: 7 levels (baselines 78/175/179/230/96/54/67). L0 one plate/gate; L1 two pairs; L2 sliding bars; L3-4 multi-plate links needing two ghosts; L5-6 bots.
- **No-effect**: noop 0.25/0.20/0.30 plus bar-only 0.23/0.20/0.31: wall/gate bumps and ACTION5 before any move; the timer drops one pixel every second action, so half the no-ops change no pixel.
- **Style**: white background, black 6-px corridors, blue player, grey ghosts, 1-px wires to comb gates, loop icons top left; ~50% maze.
- **Dynamics**: 7-9-frame slides; a rewind animates 20-65 frames; ghosts and bots move in sync with player moves.

### ka59 — kick boxes into matching frames
- **Actions**: arrows move the selected box 3 px; ACTION6 on another green box selects it (white centre dot); other clicks do nothing but cost a step.
- **Objects & interactions**: green-outlined boxes of several sizes, matching outline frames, pushable-only yellow crosses, grey walls, purple zones that block only the selected box, maroon bombs (L5+). Bumping an object kicks it: it slides 5 frames, chains, keeps sliding over purple. Bombs fill one row per move and then blast objects 9 px.
- **Win**: every frame holds a box of exactly its inner size (and crosses in cross outlines): shape-size matching, not colour.
- **Escalation**: 7 levels (baselines 28/109/51/51/33/132/326): small boxes and a purple strip; mixed shapes; crosses; crosses + barriers; bombs; everything. Board size changes per level.
- **No-effect**: 0.39/0.27/0.20, ~0.2 later; clicks 96-100% inert; arrows 0-30% blocked; from L5 arrows always tick bombs.
- **Style**: light-grey rooms (50-85%) on grey walls, native 1:1 small sprites (3-11 px), thin dark target outlines, 7-9 colours.
- **Dynamics**: kicks up to 7 frames, blasts 4-5; bombs are hidden clocks driven by moves; budget 100-200.

### lf52 — peg solitaire on islands with rail ferries
- **Actions**: click a peg to select it (ghost markers show legal landings), click a marker to jump; arrows slide all orange carriers one cell along their rails; ACTION7 undo; empty clicks deselect.
- **Objects & interactions**: islands of 6-px cells; green pegs, red from L5, blue from L8; purple jumpable-but-permanent blocks (L3+); carriers ferry pegs between islands. Jumping a same-colour peg removes it; a different colour or a purple block does not.
- **Win**: one peg left (one per colour in two-colour levels; blue not counted).
- **Escalation**: 10 levels (baselines 32→225): single board; carrier + second island; many islands, board larger than the screen; blockers; red pegs; two-colour win; blue pegs and linked carriers.
- **No-effect**: true no-op ~0, bar-only 0.98/0.54/0.40: no carriers on L0, pegs without a legal jump wiggle back, rails that do not continue. Public 0.30 is corrupted (caveats).
- **Style**: light-blue background, white-framed grids with outline and drop shadow, 4-px cross pegs; 60-70% empty; counter on row 0 changes colour every 64 actions.
- **Dynamics**: 2-3-frame actions, ~10-frame jumps, scripted camera pans to off-screen islands; hard-coded dead ends shake and show a reset button.

### lp85 — rotating rings (permutation puzzle)
- **Actions**: ACTION6 only. Red/green arrow buttons rotate one ring backward/forward by one slot. Other clicks are ignored entirely (no cost).
- **Objects & interactions**: closed rings of 2×2 tiles that share tiles at crossings, so turning one ring moves another; fixed slot markers; special yellow and (L2+) orange tiles.
- **Win**: every marker holds a tile of its kind simultaneously.
- **Escalation**: 8 levels (budgets 13-150, baselines 17/38/31/16/41/60/26/159): one ring; three interlocking rings; second tile kind; cross-shaped rings with four buttons; eight numbered forward-only rings; long snaking rings. Board size varies, auto-scaled ×2 then ×1.
- **No-effect**: 0.95/0.93/0.98 for object-pixel clicks: only 2-8% of object pixels are buttons. Effects are all-or-nothing (~300 px per effective click).
- **Style**: dark-grey board, mid-grey letterbox, many tile colours, corner-pixel markers, left-column step bar, dense rings around empty interiors.
- **Dynamics**: instant single frame; only effective clicks cost.

### ls20 — key-shaping maze with energy, lives and fog
- **Actions**: arrows move a 5×5 avatar one 5-px cell; nothing else.
- **Objects & interactions**: the carried key has shape (6), colour (4), rotation (4); stepping on a station cycles one attribute. Doors show the key they need; the wrong key is blocked (flash), the right one is consumed. Energy pickups refill and respawn; wall pushers shove the avatar to the next wall; from L4 stations patrol tracks one cell per move.
- **Win**: every door entered with a matching key.
- **Escalation**: 7 levels (baselines 22/123/73/84/96/192/186, 42 energy): rotation only; pickups; colour station + pushers; shape station + more pushers; moving station; two doors, three movers; fog (radius 20 visible).
- **No-effect**: full frame always changes (energy drops even on a bump), but 27-50% of moves leave the world unchanged (walls, wrong doors).
- **Style**: grey frame, dark walls, one-cell corridors, two-tone avatar; bottom HUD with key preview, 2-row energy bar and 3 life squares; maze ~half the screen.
- **Dynamics**: pusher shove up to 17 frames; energy out loses a life (5-frame flash, reset, refill); third life lost = GAME_OVER.

### m0r0 — mirrored twins must meet
- **Actions**: arrows move two agents at once, the right one mirrored horizontally; ACTION6 on a blue block selects it (agents go grey, arrows move the block); other clicks deselect; ACTION5 accepted but inert.
- **Objects & interactions**: each twin is blocked independently, so walls desync them; red checkered hazards blink then reset all positions (not game over, budget not refunded); coloured plates open same-coloured doors while occupied; movable blocks act as walls.
- **Win**: twins on the same cell (or swapping while adjacent), i.e. all merged.
- **Escalation**: 6 levels (baselines 30/111/203/26/500/237): symmetric room; hazards; asymmetric maze with blocks; hazards + block; three door/plate pairs; doors + plates + block. 4-agent code unused.
- **No-effect**: 0.53/0.55/0.51: ACTION5 always, clicks ~always (only blocks react), arrows 0-48% when both twins are blocked.
- **Style**: black floor, walls tinted in two colours by screen half (signals the symmetry), single-cell agents, bar on top and bottom rows; dense (35-60% floor).
- **Dynamics**: single-frame moves; hazard blink 8 frames; budget 150.

### r11l — puppets on elastic legs
- **Actions**: ACTION6 only: click a leg handle to select it, click elsewhere to teleport it there (refused if it would overlap a wall).
- **Objects & interactions**: round bodies always at the centroid of their 2-3 legs, linked by 1-px elastic lines; dotted target rings; light-blue hazard lakes revert a drop and count a hit (5 hits = GAME_OVER); from L4 white absorber bodies collect colours of paint discs they pass.
- **Win**: every body on its target; absorbers' colour sets equal their target's.
- **Escalation**: 6 levels, 60 clicks each (baselines 22/33/51/26/52/49): one body; two bodies + hazards; bigger hazards + 3 legs; 4 bi-colour bodies; absorbers + paints; more of both.
- **No-effect**: 0.86/0.62/0.52 for object-pixel clicks: wall pixels, re-selecting the handle, hazard-reverted moves. Uniform clicks on the black field would mostly be valid.
- **Style**: black field, irregular grey wall blobs, organic hazard lakes, filled discs, 5×5 cross handles (white when selected), left-column bar.
- **Dynamics**: valid move 1-3 frames; hazard hit ~20 frames of blinking then slide back.

### re86 — line-drawing overlay with paint pads and pegs
- **Actions**: arrows move the selected piece 3 px (refused only if its centre would leave the screen); ACTION5 cycles selection (white centre pixel).
- **Objects & interactions**: 1-px line pieces (plus, X, diamond, rectangle, 13-43 px); target dots; paint pads recolour a piece with a flood spreading from the contact point; pegs deform pieces (slide a crossbar, squash a rectangle) or block small ones.
- **Win**: in the composite drawing, every target dot carries its colour.
- **Escalation**: 8 levels (baselines 26→424→241): crosses; a third shape; pads; pads with 3 pieces; a peg; up to 14 pads and 2 pegs on smaller pieces.
- **No-effect**: world no-op only 0-0.07, but ≤4 px 0.18-0.31, almost all ACTION5 (a 2-px selection-dot change).
- **Style**: black background 85-94% empty (sparsest), thin bright lines, 5-14 colours, no HUD panel, bottom bar.
- **Dynamics**: recolour floods up to 27 frames; deformations permanently change geometry.

### s5i5 — colour-keyed arm chains
- **Actions**: ACTION6 only on bottom control panels: right half lengthens every bar of that colour by 3 px, left half shortens (min one unit); from L5 cross buttons rotate that colour 90°. Other clicks are inert but cost a step.
- **Objects & interactions**: 3-px bars linked in a kinematic tree (children ride parents' tips); purple walls from L1; overlapping changes are shown one frame and reverted.
- **Win**: every maroon diamond target covered exactly by a bar end.
- **Escalation**: 8 levels (baselines 20/89/106/54/162/38/86/83, budgets 50-200): 2 independent bars; 4-colour chain + walls; 5-6 colours, several targets; rotation; mixes with wall mazes; large central obstacle, 7 controls.
- **No-effect**: L0 0.77 bar-only; L1 0.39 noop + 0.30 bar-only; L2 0.51 + 0.26. Inert clicks still tick the bar, but rounding of the bar shrink makes many steps change zero pixels; also min-length shrinks and reverted collisions.
- **Style**: black ~85% empty, grey 10×6 panels with coloured hooks, 5-px diamonds.
- **Dynamics**: one frame, two when a collision is shown and undone.

### sb26 — visual programming with subroutine frames
- **Actions**: ACTION6 selects a palette item and places/swaps it in a frame slot; ACTION5 RUN; ACTION7 undo placement (no refund). No arrows.
- **Objects & interactions**: target colour sequence on top; frames (procedures) with grey slot dots, the entry frame bracketed, some slots fixed; palette items are "emit colour" or "call frame of this colour". RUN walks the program with a stack; a mismatch, empty slot or early end flashes red and resets.
- **Win**: the run emits exactly the target sequence.
- **Escalation**: 8 levels (baselines 18/28/18/19/31/23/58/18): pure ordering; second frame with fixed call; three frames; calls in the palette; repeated calls for a 9-step target; pre-filled slots; distractors; 12-step two-row target.
- **No-effect**: 0.90/0.88/0.86: clicks valid only with a selection, empty undo, failed RUN whose end frame equals the start (up to 42 animated frames).
- **Style**: dark background, 6×6 items, 1-px coloured frames, energy bar doubles as palette separator, white highlight boxes.
- **Dynamics**: 7-frame eased placement slides, 15-42-frame runs, fading highlights make even empty clicks change ~20 px; energy 64, select/undo free.

### sc25 — runes cast spells in a corridor maze
- **Actions**: arrows move 2 px (4 when large) and set facing; ACTION6 toggles 3×3 keypad slots; a matching pattern casts a spell automatically; clicking a spellbook panel demonstrates its pattern; other clicks flash the keypad red for 13 frames.
- **Objects & interactions**: resize spell (small fits narrow corridors), fireball destroying a switch and its linked barriers, teleport to a linked pad; green energy pickups; framed exit.
- **Win**: walk into the exit.
- **Escalation**: 6 levels (energy 50/25/50/35/65/60, baselines 36/6/32/83/143/50): one spell per level (resize, teleport, fireball), then combinations; automatic demo on L0.
- **No-effect**: noop 0.21/0.14/0.16 + bar-only 0.06/0.15/0.06: wall bumps (energy spent only on successful moves) and off-target clicks whose red flash reverts (40-47 of ~60 clicks per level).
- **Style**: black, grey corridors, 4-px player, spell panels down the left, 17-px keypad, vertical green energy strip.
- **Dynamics**: resize 17 frames, projectile ~10, L0 demo 22; keypad clears after each cast.

### sk48 — telescoping arm ferrying coloured blocks
- **Actions**: arrow along the arm's facing extends it one 6-px segment, the opposite retracts, perpendicular arrows slide it along a dotted rail; ACTION6 on a head switches arm (L6-8); ACTION7 undo.
- **Objects & interactions**: extending chain-pushes blocks; blocks on segments are carried; fixed receiving arms (L4); obstacles (L5); a reference arm in the bottom panel; indicator dots per correct block.
- **Win**: the colours on each paired arm, in order, equal its reference.
- **Escalation**: 8 levels (baselines 61/177/101/103/230/181/125/92): one arm and loose blocks; stacked column; carrier feeding fixed arms; obstacle; two controllable arms with 2D clusters.
- **No-effect**: 0.28/0.27/0.26: clicks 92-100% inert, arrows 0-52% (min length, off-rail, walls, blocked pushes).
- **Style**: black around a dark-grey 6-px board, hollow magenta head, checker links, bottom 10-row reference HUD, 9-13 colours.
- **Dynamics**: about half of actions take 2-3 frames; budget 196, only arrows cost.

### sp80 — liquid-flow plumbing
- **Actions**: ACTION6 selects a platform (blue), arrows move it one cell (not into top rows or cup margins), ACTION5 spills. Boards are shown rotated 0° or 180° with arrows and clicks remapped to screen directions.
- **Objects & interactions**: sources drop liquid one cell per frame; it spreads along platforms and pours off both ends; U-cups fill if entered squarely (rims split the stream); L5 corner deflectors; L6 a platform carrying a source; floor/walls are hazards.
- **Win**: all cups filled in one spill with no hazard touched.
- **Escalation**: 6 levels (baselines 39/58/25/148/96/152): board 16→20 cells, 1→3 sources, 2→4 cups, rotation varies, then deflectors, side cups, movable source.
- **No-effect**: 0.45/0.36/0.36: failed spills animate then fully restore (72-96% of spills), inert/re-select clicks, blocked moves.
- **Style**: orange background 67-86%, flat 4-px bars, no outlines, 7-9 colours, green bar that rotates with the board.
- **Dynamics**: a spill is 20-44 frames; failure flashes then reverts (edits kept); a hidden 4-failure limit, the 5th spill = GAME_OVER.

### su15 — suction pulse, Suika merges, chasing enemies
- **Actions**: ACTION6 in the playfield fires a radius-8 pulse pulling fruits and enemies toward the point (HUD clicks ignored and free); ACTION7 free undo.
- **Objects & interactions**: 9 fruit values; equal touching fruits merge upward (top two annihilate); unequal contact is illegal (16-frame flash, escalating penalty 2, 4, 6…, auto-undo). Enemies chase the nearest fruit during pulses, knock it and lower its value; equal enemies merge into tiers.
- **Win**: goal discs contain exactly the HUD recipe (fruit values, later enemy tiers); goals turn grey when unreachable.
- **Escalation**: 9 levels (baselines 22/42/26/115/36/31/8/40/41, budgets 32/48): single fruit with tutorial trail; merges; two goals; one then two enemies; enemy in recipe; wear a fruit down; tier-2/3 enemies by merging.
- **No-effect**: object-pixel clicks 87-99% (mostly HUD pixels); uniform clicks 76-97% on L0-2 (no fruit in radius), 27% on L3 (enemies move).
- **Style**: black playfield ~80% empty, dark 10-row top HUD with value ladder and recipe, fruits 1 px to 6×6, 13 colours.
- **Dynamics**: 4-5-frame pulses with a shrinking ring; enemies autonomous only during pulses.

### tn36 — binary opcodes drive a piece
- **Actions**: ACTION6 only: toggle bits in the program panel (each column one instruction), the blue button runs it, tabs load example presets on the left sandbox machine.
- **Objects & interactions**: a demo machine and a judged machine; instructions move (1-2 cells), rotate, scale or recolour the piece; walls cancel moves; toggling hazards (L5+); pads checkpoint the pose, otherwise a failed run resets.
- **Win**: after a run, the piece's position, scale, rotation and colour equal the target's.
- **Escalation**: 7 levels (baselines 32/72/26/40/30/55/62): single machine; example machine with presets; walls; scaling; recolouring; hazards with a half-speed timer. Each new preset demonstrates the new opcode.
- **No-effect**: bar-only 0.81/0.57/0.41, ≤4 px 1.00/0.97/0.86: every click shortens the timer by one pixel; bit toggles change only 1-4 px; all-zero runs end unchanged.
- **Style**: black, two 60×60 panels (grey sandbox, white editor), tab row, blue run button, timer line in row 1; dense UI.
- **Dynamics**: runs play one instruction per frame (~6-7 frames) with the active column highlighted.

### tr87 — symbol translation by rewrite rules
- **Actions**: keyboard only: left/right move a bracket cursor, up/down cycle the glyph through 7 variants of its family.
- **Objects & interactions**: a rulebook of glyph-sequence pairs, an input row and an editable output row; glyphs in 3 colour families shown in random rotations (identity matters, not rotation).
- **Win**: output equals the rule translation of the input (checked after each change).
- **Escalation**: 6 levels (baselines 54/58/40/45/71/146, budget 128/256): 1:1; 1:n; sequences; chained translation; L5 inverts the task (edit rules, input/output fixed); L6 combines editing, chaining and two-stage translation.
- **No-effect**: 0.00 everywhere: every action moves the cursor (~28 px) or swaps a glyph.
- **Style**: mid-grey rule panel, darker sequence panel, 6-px tiles on a 7-px pitch, text-like and dense, white brackets.
- **Dynamics**: single-frame edits; a colour-cycling win highlight.

### tu93 — maze with turn-based enemies
- **Actions**: arrows move one track node (6 px) if the track continues; blocked moves still cost.
- **Objects & interactions**: checkered track, player and exit; sentries dash at a player in front of them; patrollers bounce one node per turn; stalkers wake at range 2 and replay the player's moves with a delay. Entering an enemy from a safe side destroys it; an enemy reaching the player kills.
- **Win**: stand on the exit.
- **Escalation**: 9 levels (budgets 20-60, baselines 19/16/34/42/123/80/14/23/111): pure maze; sentry; 3 sentries; patroller; 4 patrollers; mix; stalker; stalker + sentry; all three.
- **No-effect**: 0.59/0.59/0.48 (L3 0.67): moving off the track; enemies act only after successful moves.
- **Style**: black, white/light-grey checkered track, tiny 3×3 sprites with a facing pixel (yellow when alerted), dense centre with big margins.
- **Dynamics**: ~7-frame moves, then enemy turns, then resolution; bursts as expanding dotted rings; up to 15-17 frames per action.

### vc33 — communicating vessels
- **Actions**: ACTION6 only: pump buttons move a partition 2-3 px along the gravity axis (one compartment grows, its partner shrinks); lit bridge buttons swap the markers of two level columns. Other clicks cost a step.
- **Objects & interactions**: white fluid columns bounded by black rods, markers riding column tops, coloured goal notches, floor stops (L5).
- **Win**: each marker level with the same-coloured notch.
- **Escalation**: 7 levels (baselines 7/18/44/61/131/34/152, budgets 50-200): gravity rotates between levels (right, left, down, down, right, left, up); board 32→64→48 (scale 2→1); pumps 2→8, markers 1→3; bridges L3, stops L5.
- **No-effect**: 92-97% of object-pixel clicks leave the world unchanged; even valid clickables fail 6-40% (pump at limit, bridge unlit). Effects when present are large (50-270 px).
- **Style**: big flat regions on grey, white fluid, black rods, 2-4-px pumps, 1-3-px markers, 42-75% background, top-row bar.
- **Dynamics**: pumps instant; swaps slide 1 px per frame, up to 43 frames.

### wa30 — hauling with helper and thief NPCs
- **Actions**: arrows turn to face then step 4 px; carrying locks facing and moves the box rigidly; ACTION5 grabs/releases the box in front, or deletes a faced thief.
- **Objects & interactions**: boxes, goal zones, walls, fences that block walkers but not carried boxes; helper NPCs BFS to free boxes and haul them to goals; thieves haul any box (even held ones) to their own zone. Outline colours encode holder and facing.
- **Win**: all boxes in goals, unheld.
- **Escalation**: 9 levels (budgets 70-200, baselines 71/119/183/98/368/68/79/442/415): solo; helper; fence; walls + five goals + three helpers; maze; thief; corridor; 2 helpers + 2 thieves + 13 boxes; everything, tight budget.
- **No-effect**: 0.22/0.00/0.11 (L5 0.17): ACTION5 with nothing in front, re-bumping a faced wall; with NPCs almost nothing is a pure no-op.
- **Style**: very light grey, full 64×64 unscaled, 4×4 entities on a 4-px grid, 85-95% empty; colour carries state.
- **Dynamics**: one frame per action, NPCs take one step after every player action.

## Measurement caveats

- **(a) lf52 and bp35 public recordings are corrupted.** `wm/dump_public.py` stores `np.asarray(r.frame[-1])` without copying; these two games later mutate that frame in place, so stored observations change after the fact. Replaying the same way gave noop 0.65 (lf52) and 0.27 (bp35) against 0.00 with real copies. Their public no-op rates (0.30 / 0.27) are mostly artefact. Fix: `np.array(..., copy=True)` before storing.
- **(b) Explorer blind spots.** ft09 public 1.00 no-op comes from the Explorer's click picker, which only targets the 48 smallest colour components and never hits the real (larger) tiles. lp85 public 1.00 because the Explorer never clicked the arrow buttons; missed clicks are free there, so not even the bar moved.
- **(c) Only the last frame is kept.** Official games return several frames per action; recordings keep `frame[-1]`. Animated responses that end where they started (failed spills and runs, rewinds, red flashes, wiggles, blinks, reverted collisions) look like no-ops even though the intermediate frames show the effect.
- **(d) Other detector effects.** ls20 public 0.00 bar-only is likely because its bar is 2 rows tall and the public bar detector missed it. In s5i5, g50t and sc25 the bar shrinks by fractional pixels, so many charged steps change zero pixels. The four notes also used slightly different random-play protocols (object-pixel vs uniform clicks, world vs full frame), so cross-game comparisons of the table's last-but-one column are approximate.

## PRINCIPLES

### Control

1. **Indirect control.** You act on X and the scored object Y responds by a rule; the goal object is rarely moved directly. Games: ka59 (kicks), sp80 (liquid), cd82 (canvas), sk48 (carried blocks), r11l (centroid), lp85 (rings), vc33 (pumps), su15 (pulse), ar25 (reflections), s5i5, ft09 L4+, tn36, sc25, lf52 (ferries), wa30 (helpers). Generator hint: make the player's controllable entity a cause (tool, lever, field, proxy) whose effect on the scored objects is computed by a deterministic rule.

2. **Coupled bodies.** One input drives several bodies under a fixed transform or link, and obstacles or constraints desync them. Games: m0r0 (mirror twins), g50t (ghost replays), r11l (body at legs' centroid), s5i5 (all bars of a colour, child bars ride parents), lf52 (all carriers move together), sp80 (global rotation remap). Generator hint: bind 2-4 bodies to one action through mirror / delayed copy / centroid / same-colour group and let walls break the coupling.

3. **Selection as a sub-action.** A click or ACTION5 chooses which entity the arrows drive; the choice is shown by recolouring (white, dot, dimming the rest). Games: ka59, sp80, m0r0, sk48 L6+, ar25, re86, cn04, r11l, sb26, lf52. Generator hint: several movable entities, one active at a time, select by click on it or ACTION5 cycling, with a 1-2-px marker change as feedback.

4. **Edit phase, then run.** Cheap edits have no world effect until a commit action plays a long deterministic simulation that is judged and reverted on failure. Games: sp80 (spill), sb26 (RUN), tn36 (run button), cd82 (pour, partly), g50t (rewind commits a ghost). Generator hint: edit phase (place tokens / move parts) + ACTION5 run that animates over many frames and restores the pre-run state on failure.

### World rules

5. **Turn-based autonomous agents and player-driven clocks.** Non-player entities follow simple policies but act only when the player acts; hidden clocks advance with moves. Games: wa30 (helpers, thieves), tu93 (sentry, patroller, stalker), su15 (chasers during pulses), ls20 (patrolling stations), g50t (bots, ghosts), bp35 (rising tide), ka59 (bombs), tn36 (toggling hazards). Generator hint: NPCs with one-line policies (BFS to target, patrol, line-of-sight dash, delayed mimic) stepped once per player action.

6. **Merge and capture arithmetic.** Equal items combine into the next tier, unequal ones conflict; captures remove same-kind items. Games: su15 (fruits, enemy tiers), lf52 (same-colour jumps), m0r0 (twins merge). Generator hint: tiered items with "equal merge up, unequal illegal" and a target count, solvable by backward generation.

7. **Order-dependent layering.** Later operations overwrite earlier ones, so the solution is a sequence, not a set. Games: cd82 (paint order), sk48 (block order on the arm), sb26 (program order), tr87 (segmentation), lp85 (non-commuting ring turns). Generator hint: operations that paint or permute over shared cells, with a target only reachable in one order.

8. **Toggle groups keyed by colour or letter.** One control acts on every object sharing its tag. Games: dc22 (letters), g50t (colour-wired plates and gates), m0r0 (plates and doors), s5i5 (colour panels), lf52 (same-colour capture), ft09 (palette cycle). Generator hint: tag objects with a colour/letter and give each tag one button or plate that cycles all members between variants.

9. **Attribute-matching locks.** A target accepts only the object whose attribute (shape, size, colour, rotation, connector kind) matches. Games: ls20 (key shape/colour/rotation vs door), ka59 (frame size), cn04 (connector kinds), re86 and vc33 (colour matching), lp85 (tile kind per marker). Generator hint: carried or movable object with 2-3 attributes modified by stations, doors displaying the required combination.

10. **Shared and attached elements.** Elements belong to several structures or ride on others, so one action moves many things. Games: lp85 (rings sharing tiles), s5i5 (kinematic tree), lf52 (pegs on ferries), sk48 (blocks on segments), vc33 (markers on columns), dc22 (carrier tiles), cn04 (one piece satisfies several connectors). Generator hint: cyclic tracks sharing cells, or parent-child attachments, so moves interfere.

11. **Deformable and editable geometry.** The world's shape is state: pieces deform, terrain is broken or toggled, objects resize. Games: re86 (pegs squash pieces, flood recolour), bp35 (break/toggle/flip/spread terrain), sc25 (resize, fireball barriers), dc22 (tile variants), tn36 (scale opcode). Generator hint: let actions change object shape or terrain cells permanently, not only positions.

### Goals

12. **Displayed reference goal.** The target is shown on screen, often away from where it applies (HUD thumbnail, recipe, outline, reference sequence, worked examples, rulebook). Games: cd82, sk48, su15, ls20, sb26, ka59, tr87, ft09 (solved examples), tn36 (demo presets). Generator hint: draw the goal as a small HUD panel or outline the agent must link to world objects, sometimes as examples from which the rule must be inferred.

13. **Derived or composite scoring.** The win checks something computed (union, overlay, set, count, local constraints), not one object's position. Games: ar25 (pieces + reflections), re86 (overlay colours at dots), su15 (counts in zones), r11l (colour sets), ft09 (equal/different neighbours), cn04 (all connectors paired), lf52 (pegs left). Generator hint: win predicates over aggregates (coverage, counts, sets, constraint satisfaction).

### Escalation

14. **One new object type per level, then combine; tutorial level 0.** Each level adds one entity or rule that modifies an existing one; the last 1-2 levels mix everything; level 0 carries a built-in hint or demo. Games: all 25 (e.g. tu93, sc25, ls20, ka59); tutorials in dc22, su15, ft09, tn36, sc25, lf52, ar25. Generator hint: a level schedule of modules (base, +A, +B, +A+B, ...) with an L0 hint that vanishes after first use.

15. **Inversion and parameter flips.** A late level flips what you edit, or a known agent gets one parameter flipped to create a new rule. Games: tr87 L5 (edit rules instead of outputs), wa30 (helper vs thief: same carry routine, opposite destination), ka59 (purple blocks only the controller), bp35 (gravity flip). Generator hint: reuse an existing mechanic with one parameter inverted (destination, editable side, gravity sign).

16. **Global transforms and partial observability.** The same rule appears under a new orientation, scale or visibility. Games: vc33 (gravity axis rotates), sp80 (board rotated with remapped controls), tr87 (random glyph rotations), lp85/tu93/vc33/dc22/ka59 (board size and scale change), ls20 L6 (fog), cn04 (grey masking), lf52/bp35 (boards larger than the screen, camera). Generator hint: per-level random rotation/mirror, scale ×1-×3, fog radius, or masking of unselected objects.

### Interface & budget

17. **Non-refilling step bar with asymmetric no-op cost.** Every game has a 1-px (or 1-column) budget bar that drains and ends in GAME_OVER; what it charges varies, and a few add refills or lives. Games: all 25; misses free in lp85, ft09, ar25, su15 HUD; select/undo free in sb26, sk48; refills/lives in ls20, sc25; half-speed timers in g50t, tn36; failure counters in sp80, r11l. Generator hint: always draw a draining bar on a border row/column, and sample per game which action classes are charged.

18. **Soft failure with auto-undo, and free undo.** A failed attempt or hazard plays an animation and rolls state back without ending the game, usually still costing budget; ACTION7 undo exists in several games. Games: m0r0 (hazard reset), sp80 (failed spill), sb26 (failed run), r11l (hazard bounce), dc22 (fall costs 20), su15 (illegal merge), s5i5 (collision revert), tn36 (piece back to start); ACTION7 in ar25, su15, sk48, sb26, lf52, bp35. Generator hint: illegal outcomes animate, revert, and charge a penalty instead of ending the level; offer ACTION7 in about a quarter of games.

19. **Most of the input space is inert; actuators are sparse and condition-gated.** Only a few small sprites react to clicks, and some appear or light up only when a condition holds. Games: vc33 (92-97% inert, bridges lit when level), dc22 (~96%, buttons hidden behind plates/keys), lp85, ka59, sb26 (slots react only with a selection), bp35, s5i5, su15. Generator hint: few small clickable actuators, some gated by state, with the rest of the screen inert.

### Visual style

20. **Sparse native-scale scenes with panels, colour encoding state.** Few mid-size drawn sprites (outlines, eyes, dots, checkers) on a large uniform background (often black; 70-95% empty), sometimes two-tone rooms; UI panels (palette, keypad, editor, HUD, control strip) take 20-50% of the screen; 1-px lines and markers; state shown by recolouring (selected = white, held, satisfied, alerted); scale changes between levels. Games: re86, s5i5, wa30, cn04 (sparse); m0r0 (two-tone walls); tn36, sc25, s5i5, ft09, dc22, su15 (panels); wa30, cn04, tu93, r11l, sb26 (recolouring). Generator hint: render at native scale with 3-11-px detailed sprites, a dominant flat background, a side or top panel, and colour changes as the main state signal.

### Dynamics

21. **Multi-frame responses.** One action can produce 2-65 frames (slides, falls, floods, runs, rewinds, flashes, blinks, camera pans), and the last frame may equal the first. Games: g50t (rewind 20-65), sp80 (20-44), sb26 (up to 42), vc33 (up to 43), re86 (27), bp35 (falls ~20, camera), lf52 (pans), ls20 (17), tu93 (15-17), sc25, dc22, su15. Generator hint: emit intermediate frames for every animated effect, including failures that revert, and occasionally move the camera.

## Escalation patterns

Rules escalate by adding one module per level and recombining them, rarely by just enlarging the board:
- tr87: 1:1 rules → 1:n → sequence:sequence → chained → edit the rules (inversion) → all combined.
- tu93: pure maze → sentry → patroller → stalker → all three mixed.
- sc25: one spell per level (resize, teleport, fireball) → pairs → all three.
- wa30: solo → helper → fence → walls → thief (same routine, flipped destination) → helpers + thieves.
- vc33: same pumps, but the gravity axis rotates every level and bridges (L3) and stops (L5) are added.
- cn04: 2 → 4 pieces, then masking hides unselected connectors (L2), then variant stacks and pre-rotation.

Difficulty, measured by human baseline actions, grows strongly in many games: ka59 28→326, dc22 59→578, re86 26→241, lf52 32→225, bp35 21→163, m0r0 30→237 (500 on L5), i.e. roughly 5-10× from level 0 to the last level. It is not universal: cd82, su15, sc25, sb26 and tn36 stay roughly flat, their difficulty coming from new rules rather than longer solutions.

## Implications for the generator

- **Rule curriculum inside each game**: 6-10 levels, one new object type or rule per level, combinations in the last levels, a tutorial affordance in level 0 that disappears after first use, baselines growing several-fold.
- **Realistic inertia**: under exploration roughly 1/3 of actions (more in click-heavy games) should have no visible effect, coming from inert clicks on decor, blocked moves, selection-only actions, failed runs that revert, and gated actuators; the bar should still move for charged actions.
- **Indirect and coupled control by default**: prefer proxies (tools, programs, ghosts, levers, colour groups) over a single avatar pushing the goal object; avatar navigation is over-represented in our families (10/16 vs 9/25).
- **Displayed goals**: show the target as a HUD reference, outline, recipe or worked example, and make the win predicate an aggregate (coverage, counts, sets, constraints) in a good share of games.
- **Sparse native rendering**: flat backgrounds covering ~70-95% of the screen (often black), small detailed sprites, thin 1-px lines, UI panels where relevant, recolouring for state, scale and orientation changes between levels.
- **Budget interface**: always a draining bar on a border; sample which actions it charges, add lives/refills or failure counters in some games, soft failures with auto-undo, ACTION7 undo in about a quarter of games.
- **Multi-frame output**: generate intermediate frames for animated consequences (and record all of them, copying each frame), so a world model can learn effects that leave the final frame unchanged.
- **Turn-based autonomy**: include simple NPC policies and player-driven clocks that step once per action, with ally/enemy variants obtained by flipping one parameter.

## Coverage by the v4 generator (`gen/v4/`, `gen/families_v4.py`)

Each v4 family is our own procedural design (no official code, layout or sprite is reused); it realises one or more
of the principles above, has >= 2 variants, a short tutorial level 0 and one new rule per level (order drawn per game).

| principle | v4 families |
|---|---|
| 1 indirect control | kick4 (kicked blocks slide), pump4 (pumps move floats), flow4 (deflectors guide a stream), merge4 (pulse attracts), bary4 (body at the centroid of handles), arm4 (arm carries blocks), cover4 (reflections cover), fall4 (reshape terrain, gravity moves you) |
| 2 coupled bodies | twins4 (mirror / inverse / rotated bodies), bary4, ghost4 (past runs replayed) |
| 3 selection as a sub-action | kick4 (2nd pilot), flow4, cover4, lines4, bary4, paint4 (ink / tool) |
| 4 edit phase, then run | program4 (tokens then ACTION5 run, failed run reverts), flow4 (move deflectors then release, failed run reverts), ghost4 (ACTION5 commits a run) |
| 5 turn-based agents, clocks | npc4 (patroller, turret, delayed mimic, rail dog; helpers vs thieves), keyattr4 (moving station), remote4 (carrier), ghost4 (timed gates) |
| 6 merge arithmetic | merge4 (equal tiers merge, recipe) |
| 7 order-dependent layering | paint4 (layers), arm4 (pick order), program4 (instruction order), rotor4 (non-commuting turns) |
| 8 toggle groups by colour | remote4 (bridge groups), ghost4 (plate -> gate of its colour), twins4 (plate / door) |
| 9 attribute-matching locks | keyattr4 (shape / colour / rotation key vs doors), kick4 (colour-keyed frames), flow4 (colour-keyed cups), pump4 (float colour vs notch) |
| 10 shared elements | rotor4 (tracks share cells), pump4 (pipes), arm4 (blocks ride the arm) |
| 11 editable geometry | fall4 (break / build blocks, gravity flip), lines4 (paint pads recolour, pegs) |
| 12 displayed goal | all: HUD thumbnails (paint4, rotor4), recipes (merge4), reference sequences (arm4, program4), rule tables (rewrite4), worked examples (infer4), outline frames (kick4, cover4), notches (pump4), key requirements on doors (keyattr4) |
| 13 derived scoring | cover4 (union of pieces + reflections), lines4 (overlay colours at dots), infer4 (local constraints), merge4 (tier in zone) |
| 14 one rule per level | all families (`lvl["mods"]`, `rules_per_level` in games.json) |
| 15 inversion / flips | rewrite4 (edit the rules instead of the answer), npc4 (helper vs thief), fall4 (gravity flip), rotor4 (reverse button) |
| 16 global transforms | pump4 and flow4 (rotated scene / gravity), rewrite4 (rotated runes), paint4 (rotated thumbnail), sizes and scale ×1/×2 per game |
| 17 step bar, asymmetric cost | base V4Game: proportional 1-px bar full at level start, mostly at the bottom; `free_miss` (missed clicks free) in click families; budgets sometimes > bar length (many steps change no pixel) |
| 18 soft failure / undo | program4, flow4 (failed runs revert), twins4 and bary4 (hazards revert), ACTION7 undo in ~30 % of games (core) |
| 19 inert input space | inert decor, panels and glyphs in remote4, rotor4, pump4, program4; buttons lit only under conditions (remote4 plate) |
| 20 sparse native scenes | base V4Game: native ×1 (sometimes ×2) full-area rendering, role palettes (`v4/draw.py`), two-tone rooms, solid drawn sprites |
| 21 multi-frame responses | NOT covered: our API returns one frame per action; failed runs simply leave the frame unchanged |

Not (or barely) covered: camera scrolling / boards larger than the screen (lf52, bp35), fog (ls20 L6), spell drawing on a keypad (sc25), binary opcode decoding (tn36), connector jigsaws (cn04).
