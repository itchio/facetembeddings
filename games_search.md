# Games Search Table Reference

The `games_search` contains all indexed content on itch.io indexed for
filtering. This document is a summary of the most common
facet types and values.

## Table Schema

```sql
CREATE TABLE games_search (
  game_id integer NOT NULL,
  facets tsvector NOT NULL,
  -- ... other internal columns
)
```

## Facets

The `facets` column contains a space-separated list of prefixed tokens
representing searchable game attributes. Each facet follows the pattern
`'prefix.value'` (with quotes in the tsvector representation).

### Facet Format

Facets are queried using PostgreSQL's `@@` operator with tsquery strings:
```sql
-- Windows games tagged with horror
WHERE facets @@ 'p.windows & tg.horror'::tsquery

-- Free games OR games with demos
WHERE facets @@ 'm.free | m.demo'::tsquery

-- macOS projects that are NOT games
WHERE facets @@ 'p.osx & !c.1'::tsquery
```

> Note: some facets have negative variants to avoid using negative filters, eg.
> `n.yes` and `n.no` for NSFW `true` or `false`

### Facet Categories

Based on production data analysis (20,000 game sample):

| Prefix | Category | Unique Values | Cardinality |
|--------|----------|---------------|-------------|
| `tg` | Tags | 1,602+ | High |
| `p` | Platform | 7 | Low |
| `m` | Monetization/Price | 5 | Low |
| `c` | Classification | 9 | Low |
| `n` | NSFW Status | 2 | Low |
| `j` | Game Jam | 2 | Low |
| `t` | Type | 5 | Low |
| `r` | Release Status | 5 | Low |
| `tl` | Made With (Tool ID) | 44+ | Medium |
| `in` | Input Methods | 25+ | Medium |
| `ad` | Average Duration | 6 | Low |
| `ln` | Languages | 29+ | Medium |
| `ay` | Accessibility | 8 | Low |
| `cl` | Code License | 14 | Low |
| `al` | Assets License | 8 | Low |
| `tt` | Tool Tags | 10+ | Low |
| `pc` | Player Count | 8 | Low |
| `mp` | Multiplayer | 3 | Low |
| `ft` | Featured (flag) | 1 | Low |

### Low-Cardinality Facets (Complete Values)

#### Classification (`c`)
- `c.1` - Game (most common)
- `c.2` - Assets
- `c.3` - Game mod
- `c.4` - Physical game
- `c.5` - Soundtrack
- `c.6` - Other
- `c.7` - Tool
- `c.8` - Comic
- `c.9` - Book

#### NSFW Status (`n`)
- `n.no` - Not NSFW (vast majority)
- `n.yes` - Mature/NSFW content

#### Game Jam (`j`)
- `j.no` - Not part of a game jam
- `j.yes` - Submitted to one or more game jams

#### Type (`t`)
- `t.1` - Downloadable
- `t.2` - Flash (legacy)
- `t.3` - Unity Web Player (legacy)
- `t.4` - Java applet (legacy)
- `t.5` - HTML5/browser game (most common)

#### Monetization (`m`)
- `m.free` - Free/PWYW game (dominant majority)
- `m.paid` - Paid game (any price)
- `m.500` - Priced ≤$5.00
- `m.1500` - Priced ≤$15.00
- `m.demo` - Has demo available

#### Release Status (`r`)
- `r.1` - Released (most common)
- `r.2` - In development
- `r.3` - On hold
- `r.4` - Prototype
- `r.5` - Canceled

#### Platform (`p`)
- `p.web` - Browser-playable (very common)
- `p.windows` - Windows
- `p.osx` - macOS
- `p.linux` - Linux
- `p.android` - Android
- `p.ios` - iOS
- `p.mw` - Mobile web/phone browser

#### Multiplayer (`mp`)
- `mp.1` - Local multiplayer
- `mp.2` - Server-based network multiplayer
- `mp.3` - Ad-hoc network multiplayer

#### Player Count (`pc`)
- `pc.2` through `pc.8` - Specific player counts (2-8 players)
- `pc.9p` - 9 or more players

#### Average Duration (`ad`)
- `ad.1` - A few seconds
- `ad.2` - A few minutes (most common)
- `ad.3` - About a half-hour
- `ad.4` - About an hour
- `ad.5` - A few hours
- `ad.6` - Days or more

#### Accessibility (`ay`)
- `ay.1` - Colorblind-friendly
- `ay.2` - Subtitles
- `ay.3` - Configurable controls
- `ay.4` - High contrast
- `ay.5` - Interactive tutorial
- `ay.6` - One button
- `ay.7` - Blind-friendly
- `ay.8` - Textless

#### Code License (`cl`)
- `cl.1` - Proprietary/All rights reserved (most common)
- `cl.2` - Apache License 2.0
- `cl.3` - MIT License
- `cl.4` - BSD 2-Clause
- `cl.5` - BSD 3-Clause
- `cl.6` - GPL v2
- `cl.7` - GPL v3
- `cl.8` - Eclipse Public License
- `cl.9` - Artistic License
- `cl.10` - LGPL v2.1
- `cl.11` - LGPL v3
- `cl.12` - Mozilla Public License
- `cl.13` - Unlicense
- `cl.14` - zlib License

#### Assets License (`al`)
- `al.1` - Proprietary/All rights reserved (most common)
- `al.2` - Creative Commons Zero (CC0)
- `al.3` - CC BY 4.0
- `al.4` - CC BY-ND 4.0
- `al.5` - CC BY-SA 4.0
- `al.6` - CC BY-NC 4.0
- `al.7` - CC BY-NC-ND 4.0
- `al.8` - CC BY-NC-SA 4.0

#### Binary Flags (no suffix)
- `ft` - Staff-featured game (rare)

### Medium/High-Cardinality Facets

#### Tags (`tg`) - 1,600+ unique values
User-applied tags, slugified. Most common examples:
- `tg.action` - Action games
- `tg.platformer` - Platform games
- `tg.puzzle` - Puzzle games
- `tg.2d` - 2D games
- `tg.shooter` - Shooter games
- `tg.arcade` - Arcade games
- `tg.adventure` - Adventure games
- `tg.pixel-art` - Pixel art style
- `tg.horror` - Horror games
- `tg.rpg` - Role-playing games
- `tg.simulation` - Simulation games
- `tg.strategy` - Strategy games
- `tg.visual-novel` - Visual novels
- `tg.retro` - Retro-styled games
- `tg.casual` - Casual games

Tags are the most diverse category and drive most browse filtering.

#### Made With / Tools (`tl`) - 44+ tool IDs
References tool IDs from the `tools` table:
- `tl.3` - Unity
- `tl.14` - Godot
- `tl.16` - GameMaker
- `tl.7` - Construct
- `tl.34` - Unreal Engine
- `tl.11` - RPG Maker
- `tl.5` - Twine
- `tl.13` - Ren'Py

#### Tool Tags (`tt`) - 10+ tag IDs
References `tool_tags` table for categorizing the tools/engines used.

#### Languages (`ln`) - 29+ ISO language codes
- `ln.en` - English (dominant)
- `ln.es` - Spanish
- `ln.fr` - French
- `ln.de` - German
- `ln.it` - Italian
- `ln.ja` - Japanese
- `ln.pl` - Polish
- `ln.ru` - Russian
- `ln.pt` - Portuguese
- `ln.ko` - Korean

#### Input Methods (`in`) - 25+ types
Most common:
- `in.1` - Keyboard
- `in.2` - Mouse
- `in.3` - Xbox controller
- `in.4` - Gamepad (generic)
- `in.5` - Joystick
- `in.6` - Touchscreen
- `in.7` - Voice control
- `in.8` - Oculus Rift
- `in.13` - Accelerometer
- `in.15` - Phone (as controller)
- `in.16` - Dance pad
- `in.20` - PlayStation controller
- `in.21` - MIDI controller

