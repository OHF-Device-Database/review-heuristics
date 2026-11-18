# OHF Device Database: guidelines on user contributions (text)

Scope: Text side only, English only. Translations and images/artifacts are separate topics.

## 1. Purpose of this document

User contributions to the Open Home Foundation Device Database (DDB) will include free text next to the structured capability data received from Home Assistant submissions.

This document address three things:

- Collects good and bad examples for the four text sections (name, summary, getting going, quirks), with the reason for each verdict.
- Proposes a first skeleton of "how to write device information for the Device Database", kept short so the entry barrier stays low.
- Lists the capabilities the contribution review bot needs on the text side, derived from the writing rules.

## 2. What the text has to add on top of the structured data

The Preview Edition already shows, per device: name, manufacturer, category, connectivity (local connection or requires internet), Home Assistant integration, supported entities, ecosystems, software and hardware version, model number, real-world telemetry (report count, first seen, "also listed as") and where-to-buy links. See for example Shelly Plus 1PM (SNSW-001P16EU).

Additional free text is only worth having when it answers something the structured fields cannot:

- What this device physically is and does, in one glance (summary).
- What a person has to do between unboxing and seeing it in Home Assistant (or any other supported ecosystem for that matter), beyond "Settings > Devices and services > Add integration" (getting going).
- What will surprise, limit, or mislead them once it runs (quirks).

Any sentence that plainly restates a structured field ("connects locally", "supports the sensor and switch entities", "model number SNSW-001P16EU") is noise, unless it adds additional value.

The general voice is already set by the Editorial stance:
Neutral and factual, wiki editor voice, verbs over adjectives, uncertainty stated explicitly, no recommendations and no rankings.
Everything below is that stance applied to four concrete fields.

## 3. The four sections at a glance

| Section | Answers the question | Form | Length guide |
| --- | --- | --- | --- |
| Name | What is this product called, so I can match it to the box and to other databases? | Marketing name as the brand writes it; a leading brand may be omitted where the name stands on its own (Roomba Combo j9+). Model number in parentheses when it exists; the parentheses are the displayed form, the identifier itself is a structured field.<br><br>Search should match the marketing name with and without brand, plus all identifiers (aliases). | One line |
| Summary | What is it and what does it do, physically? | One or two plain sentences, no bullets | Up to about 200 characters |
| Getting going | What do I have to do, in which order, to get it into Home Assistant? | Ordered steps, each one action, links to the integration docs where they cover it | Three to eight steps; if it needs more, e.g. link out. See section 7 for more information. |
| Quirks | What will surprise me, or not work, once it runs? | Bullet list, one fact per bullet, each with how it was observed or sourced | One line per bullet, as many bullets as there are quirks |

## 4. Good and bad examples

Examples marked with (*) are written for this exercise and have not been verified against the device. They illustrate style, not device facts, and are a good place for people who own the device to correct the content in a comment. The Roomba examples come from the text samples sheet.

### 4.1 Name

Verdicts in this table: Good = preferred form, Acceptable = allowed but not preferred (the bot accepts both), Bad = rejected.

Duplication checks are essential upon adding / changing device names.

| Example | Verdict | Why |
| --- | --- | --- |
| Roomba Combo j9+ | Good | The name as iRobot writes it, nothing added. Matches the box and other lists. |
| iRobot Roomba Combo j9+ | Acceptable | Roomba is the product line (see row above), iRobot is the brand and lives in the brand field. |
| Shelly Plus 1PM (SNSW-001P16EU) | Good | Preferred form: this is how Shelly writes the product name. The brand appearing in both the name and the brand field is fine; display and search must dedupe it. |
| Plus 1PM (SNSW-001P16EU) | Acceptable | Acceptable: the brand is dropped and the name still stands on its own; the brand field carries it. |
| Philips Hue motion sensor (SML001) | Good | Signify prints the name that way. |
| Hue motion sensor (SML001) | Acceptable | Acceptable: the brand is dropped and the name still stands on its own; the brand field carries it. |
| Roomba Combo® j9+ | Bad | Trademark symbols are not part of a name and break search and matching. |
| Roomba Combo j9+ Robot Vacuum and Mop | Bad | Repeats the category (robot vacuum) and a capability (mop) that live in structured fields. |
| Philips Hue Motion Sensor SML001 Zigbee | Bad | Manufacturer, protocol and model number crammed into the name. This is how source sites such as blakadder title their pages, so ingestion will produce it |
| Temperature, Humidity and Pressure Sensor | Bad | A description, not a name. Several manufacturers sell a product with exactly this title; it needs the model number (WSDCGQ11LM) to be identifiable. |
| SNSW-001P16EU | Bad | Model number alone. Correct but unrecognisable to a person holding the box. Use it in parentheses after the product name.<br><br>Bad as a contributed name; acceptable as the generated placeholder from a submission until someone contributes the marketing name. |
| Best Zigbee plug 2026!! | Bad | Not a name at all. Opinion, no manufacturer, no model. |

### 4.2 Summary

| Example | Verdict | Why |
| --- | --- | --- |
| Vacuum and mop combo bundled with dock for dust bin collection and water supply. | Good | Says what is in the box and what the dock does. Nothing a structured field already says. |
| Single-channel Wi-Fi relay with power metering for in-wall installation behind an existing switch. (*) | Good with caveat | Physical form (in-wall, behind a switch), what it switches (one channel), what it measures. Reads in one breath. Protocol would better be captured as a trait. |
| Battery-powered Zigbee sensor reporting motion, light level and temperature; indoor version of the Hue outdoor sensor. (*) | Good with caveat | Power source, measured values, and a disambiguation against a sibling product. That last part is exactly the kind of thing structured data cannot carry.<br><br>Protocol would better be captured as a trait. |
| Zigbee USB stick for Home Assistant that runs either Zigbee or Thread firmware, not both at once. (*) | Bad | Names an ecosystem in the summary; summaries stay neutral. The firmware constraint is real but is a quirk, not what the device is. |
| USB-connected Zigbee and Thread radio. (*) | Good | Physical form and what it is, nothing else. The either/or firmware limitation moves to quirks (see 4.4). |
| The Roomba Combo® j9+ robot vacuum and mop gives your home a powerful, hands-free clean. Dirt Detective, powered by iRobot OS, will automatically clean the dirtiest rooms first. While the Clean Base® Auto-Fill Dock makes sure your robot always has what it needs to get the job done. | Bad | Vendor copy. Adjectives instead of facts, product feature names used as if they explained something, trademark symbols. This is also what the CSA product list carries as "description", so ingestion needs a rewrite step, not a copy step. |
| Great little relay, works perfectly with HA, highly recommended. | Bad | Opinion and recommendation. The editorial stance rules both out. Says nothing about what the device is. |
| Shelly Plus 1PM, model SNSW-001P16EU, connects locally and exposes sensor, switch and update entities via the Shelly integration. | Bad | Every clause repeats a structured field on the same page. Zero added value. |
| Smart home device. | Bad | Too short to say anything. The category field already says more. |
| This is a Zigbee 3.0 temperature and humidity sensor with pressure measurement that uses a CR2032 coin cell which lasts about two years in my experience, pairs with ZHA and Zigbee2MQTT and deCONZ, has been sold since 2017 and was superseded by the T1 version which uses a different battery and has a slightly different housing with a magnet mount. | Bad | Correct facts, wrong place. Battery life belongs in quirks, pairing in getting going, the successor product in a related-devices field. One sentence, 60 words, no reader finishes it. |
| I think it is a motion sensor? | Bad | Uncertainty is allowed, but phrased as a question it is a request for help, not information. Write "Motion sensor (type unverified, please confirm)" if you are not sure. |

### 4.3 Getting going

The default path for most devices is one line:

"Add it through the [integration name] integration; Home Assistant discovers it automatically."

A getting going section only earns its space when something has to happen outside that path.

| Example | Verdict | Why |
| --- | --- | --- |
| Pairing with Zigbee: put the coordinator in pairing mode, then press and hold the button on the sensor for about five seconds until the LED blinks blue. If pairing stalls, keep pressing the button once per second until the device shows up.<br><br>Pairing with Thread: put the coordinator in pairing mode, then press and hold the button on the sensor for about five seconds until the LED blinks blue  (*) | Good | Names the context (direct Zigbee / Thread pairing for device that supports both), gives the physical action, the visual confirmation, and the known workaround. Adapted from the blakadder pairing instructions, which is the kind of content we want to ingest. |
| 1. Power the relay and join it to your Wi-Fi with the Shelly app or via its own access point (shellyplus1pm-XXXX). 2. Home Assistant discovers it; confirm the Shelly integration dialog. 3. If it is not discovered, add the Shelly integration manually with the device's IP address. (*) | Good | Steps in the order they happen, one action each, with the fallback last. Points to the app only for the step that needs it. |
| Starting jobs in specific rooms outside the app (the Roomba example from the sheet, abridged): 1. In the iRobot integration, open Configure and untick Continuous so Home Assistant releases the connection. 2. Start a room job in the iRobot app, then close the app. 3. Read the room id with dorita980 (command in the linked repo). 4. Send vacuum.send_command with the pmap_id and region_id. | Good, with a caveat | Precise, reproducible, and every step is needed. It also shows the limit of the field: forty lines with a bash one-liner and a YAML block. The right form is a short step list in the DDB plus a link to a community guide or the integration docs for the code, so that the DDB does not become a second copy of a tutorial that will drift out of date.<br><br>See section 7 for further details. |
| Just add it in HA, works out of the box. | Bad | Not wrong, but says nothing. If discovery works, leave the field empty and let the default one-liner show. |
| Download the app, create an account, follow the instructions, then it shows up. | Bad | Vague ("follow the instructions") and hides a fact that matters: the device needs a manufacturer account. That belongs in quirks as a stated requirement, and the steps should say which app and what to do in it. |
| Took me three evenings to get this working, the trick is to reset it twice and then it finally pairs. Not sure why. | Bad | Anecdote without a reproducible step. Rewrite as "Pairing may need a second factory reset (reported by several users, cause unknown)" and put it in quirks. |
| sensor: - platform: "mqtt" state_topic: "zigbee2mqtt/[FRIENDLY_NAME]" ... | Bad | Manual MQTT YAML for a device that MQTT discovery handles. Legacy configuration found on source sites; the bot should flag YAML and code for human review and suggest linking to the integration documentation instead. |
| Go to Settings > Devices and services, click Add integration, search for Shelly. | Bad | Repeats the generic text the device page already shows for every device. |

### 4.4 Quirks

| Example | Verdict | Why |
| --- | --- | --- |
| Starting jobs in specific rooms requires a complex setup. / Job settings for suction, liquid amount, SmartScrub and cleaning passes can only be configured in the app. | Good | One fact per bullet, states a limit of the integration against the app, no judgment. |
| Known to drop off the network when routed through some non-Aqara Zigbee routers; re-pairing near the coordinator usually fixes it. (*) | Good | States the problem, the condition under which it happens, and the fix. Hedged where the community knowledge is hedged ("some", "usually"). |
| Runs either Zigbee or Thread firmware, not both at once; switching needs a firmware flash. (*) | Good | A limitation the buyer must know and the connectivity traits cannot express. Moved here from the summary (see 4.2). |
| Shows "Local connection" but initial setup and the voice assistant need a Google account and internet access; only Cast playback and volume work locally. (*) | Bad | Contradicts a structured field. |
| Use the USB extension cable from the box; plugged straight into a USB 3 port the stick suffers interference and devices drop. (*) | Good | Concrete cause, concrete remedy. |
| Temperature entity is reported to be about 1 to 2 °C high when mounted on a wall near a light switch (unverified, source wanted). | Good | Uncertainty marked exactly as the editorial stance asks. |
| Terrible battery life, avoid. | Bad | Opinion plus recommendation. Rewrite as "CR2032 battery lasts around six months with default reporting interval (owner reports)". |
| Works great, no quirks! | Bad | Leave the field empty instead. |
| Does not work with ZHA. | Bad | Absolute claim with no version, no date, no source. ZHA support changes with releases. Write "Not supported by ZHA as of Home Assistant 2026.10 (see issue #1234)". |
| Requires internet. | Bad | Already a structured field. If the point is which features need internet, say that. |
| Had issues after the update, support was useless, iRobot should fix their API. | Bad | Complaint aimed at the vendor, no device fact a reader can act on. |
| Motion sensor, Zigbee, battery powered, pairs via ZHA. Sometimes slow. | Bad | Four facts that belong in summary or structured data, then one actual quirk too vague to use ("slow" at what, under which conditions?). |

## 5. Skeleton: how to write device information for the Device Database

This is deliberately short. It should fit on one screen and be referenced e.g. next to the contribution form. Everything in sections 3 and 4 is the long version.

### 5.1 One rule

Write what the device does, in plain words, so that someone holding the box and someone reading the page both recognise it. Not what you think of it.

### 5.2 One recipe per field

- **Name:** the product name as the manufacturer prints it, then the model number in parentheses. You may drop a leading brand where the name stands on its own, since the brand has its own field. Do not add anything the manufacturer does not print: no category, no protocol, no symbols.
- **Summary:** one or two sentences. What it is, what it does, what is in the box. If you can delete a word without losing a fact, delete it.
- **Getting going:** only the steps that are not "add the integration". One action per step, in order. Link to the integration documentation instead of copying it.
- **Quirks:** one surprise per bullet. Say when it happens and, if known, how to work around it. If you are not sure, say so in the bullet.

### 5.3 The five-point check (what makes a section "nice" to read / consume)

A section passes when all five hold. These are also the checks the review bot runs.

- **Adds something:** It says at least one thing the structured fields on the page can't express.
- **Factual:** Verbs and nouns, no adjectives of judgment, no "great", "terrible", "recommended", "avoid".
- **Specific:** Names the condition, the version, the button, the entity. No "sometimes", "somehow", "just", "etc." without a fact next to it.
- **Honest about certainty:** Unverified claims are marked as such, ideally with a source or a request for one.
- **Short and in the right field:** Within the length guide of section 3, and each fact sits in the field that asks for it.

### 5.4 Things that never go in

- Vendor marketing copy, including copied product descriptions from manufacturer sites or product listings
- Recommendations, rankings, comparisons with other products ("better than the Aqara one"). That is something we might tackle at a later point in time in a structured manner.
- Complaints of any kind, e.g. about the manufacturer or support.
- Prices, shop links, affiliate links ("where-to-buy" is a separate, curated feature).
- Personal data: names, email addresses, IP addresses, serial numbers, account ids, API keys, Wi-Fi credentials.
- Text in a language other than English, for now. Translations are a separate topic.

## 6. What this means for the review bot (text side)

Each row maps a writing rule to a capability and to the action the bot should take.

- **Accept:** means the contribution will be merged
- **Reject:** means the contribution does not enter review but changes are requested together with an (actionable) reasoning. In a first step link to the guidelines for reference.

For a potential later step and not to be further discussed now but kept as reference is:

- **Suggest:** means the bot proposes a rewrite the contributor can accept.

The guiding principle for DDB contributions is that OHF has no capacity to monitor content, so the bot should resolve as much as possible with the contributor before a human sees it.

### Rules

- Name adds nothing beyond the manufacturer's own name: no symbols, no category, no protocol, no brand the manufacturer does not print; a leading brand may be dropped
- Name carries a model number when the device has one
- Text adds something beyond structured data
- Factual, no opinion or recommendation
- No vendor marketing copy
- Specific, not vague
- Uncertainty marked
- Length and form per field
- No duplicate of generic Home Assistant steps
- Code and configuration blocks
- No personal data or secrets
- No prices or shop links
- Links are allowed only to documentation and sources
- English only
- Not abusive, not spam
- Consistent with ingested sources

Two things the bot should not do:

- verify device facts it has no source for (it cannot own the device)
- rewrite without the contributor's consent

## Appendix

### B. How other community databases handle this

Three projects with user-contributed data and a long track record were checked: MusicBrainz (structured music metadata with free-text annotations), Wikipedia together with Wikidata (prose articles and one-line descriptions), and OpenStreetMap (structured map data). All three face the same tension as the DDB: anyone can contribute, maintenance capacity is small, and consistency still has to hold. The patterns below are the ones that transfer; section B.4 says what to take over and what not.

#### B.1 MusicBrainz

- **Layered guidelines.** One short Style Principles page sits on top: usually follow the style guidelines; if sources are inconsistent, use the most common version; if the artist intended a special spelling, follow artist intent. Below it are detailed per-entity guidelines (titles, artist, release, recording, relationships) and language-specific pages for capitalisation. A newcomer needs the top page; an editor who hits an edge case has somewhere to go.
- **Burden of proof on the deviation.** Whoever claims that a deviation from the guidelines is artist intent has to prove it. The default wins unless there is evidence. For the DDB this becomes: the name is what the manufacturer prints, and whoever wants something else brings a source.
- **Structured data first, free text second.** Annotations exist for what does not fit the schema: additional facts, longer disambiguation, credits the schema cannot hold (they call the last one Sub Optimal Credits). Once a structured field exists for the information, it moves out of the annotation. This is the same rule as section 2 of this document, and it also gives a lifecycle: a quirk that turns out to be common across devices becomes a candidate for a structured field.
- **Edits are proposals, reviewed against the guidelines.** Trivial classes of edits (capitalisation, punctuation) apply immediately for everyone. Most other edits are open for voting and apply after seven days if nobody votes no. Trusted auto-editors can approve immediately. Anyone voting no must say why in an edit note. The guidelines are explicitly the yardstick that voters judge by.
- **Legacy data is acknowledged.** An Old style practices page explains that a large database lags behind guideline changes, so editors will meet old patterns and may fix them. Ingested blakadder or CSA text in the DDB will be in that situation from day one.
- **Changing a guideline has a process:** discussion in the forums, then a ticket in a style issue tracker. Nobody edits the guidelines on their own.

#### B.2 Wikipedia and Wikidata

- **A pyramid of rules, with the top small enough to read in a minute.** Five pillars, then the Simplified ruleset, then the full Manual of Style. The policy page itself says there is no need to read any policy to start editing. Each guideline page opens with a box called This page in a nutshell. Our section 5 plays the role of the nutshell; sections 3, 4 and 6 are the layers below it.
- **Three content policies carry everything else:** neutral point of view, verifiability, no original research. The DDB editorial stance already has the first two in spirit. The third is where the DDB deliberately differs: owner observations are original research and we want them, so the DDB replacement is to state the basis of each claim (observed on my device, from the documentation, from an issue) rather than to forbid first-hand knowledge.
- **Words to watch, not forbidden words.** The Manual of Style keeps lists of peacock terms (great, legendary, best), weasel words (some people say, it is widely known), editorialising adverbs (notably, interestingly) and vague terms (sometimes, somehow, at some point). The page opens by saying that no word is forbidden; the lists are cues to look closer. That is the right posture for a bot too: these lists are a trigger for a suggestion, not a block.
- **Tag the defect, keep the content.** Citation needed, weasel, by whom and similar maintenance tags leave the text in place, make the problem visible to readers, and let anyone fix it later. This is the model for our unverified, source wanted marker and for how flagged DDB content can stay public while the community acts on it, which fits the OHF principle of minimal moderation overhead.
- **Automation in three layers.** Edit filters run before the edit is saved: pattern rules that can warn, tag or disallow, maintained by a small group of filter managers. Vandalism bots (ClueBot NG) run after the save with a machine-learning classifier and revert. Humans with semi-automated tools handle the rest. The extension was renamed from abuse filter to edit filter because false positives were alienating good-faith editors. Lesson for the DDB: do the checks before submission, explain every hit, and treat a false positive as a bug in the rule.
- **Wikidata descriptions are close to a spec for our summary field:** short, usually lowercase, no initial article, not a definition, purpose is to disambiguate, avoid information likely to change, avoid opinionated or promotional wording. Tooling warns when a description is too long or starts with an article. The rules that survived are the machine-checkable ones.
- **Wikipedia short descriptions can be set to none** when the title alone identifies the subject. That is a deliberate empty state, not a missing value.

#### B.3 OpenStreetMap

- **One Good practice page,** reorganised in 2019 into seven headings after it had grown to 22 unsorted sections. The first line is to be bold, edit the map; corrections are expected and reverts are cheap.
- **Verifiability as the central rule:** map what is on the ground, do not map temporary or historic things, do not map local legislation unless an object represents it. The test is whether another mapper could check the fact. For quirks this is a good test: a quirk should describe something another owner of the same device could reproduce or observe.
- **Do not map for the renderer:** never enter wrong data to make the map look right. The DDB version is do not write for the bot or the page layout; if the field does not fit the fact, raise it instead of bending the fact.
- **Mark estimations with fixme:** an explicit, searchable marker for uncertain data, which others can query and resolve. Same role as our unverified marker.
- **Do not copy from other maps:** licensing of sources is a contribution rule, not only a legal footnote. Ingesting blakadder and CSA content needs the operators' approval for the same reason, and copied vendor copy is a licensing problem on top of a style problem.
- **Document your custom tags:** if you need something the schema does not have, write it down where others can find it. Equivalent for us: a quirk pattern that recurs should be proposed as a structured field, not reinvented in free text per device.

#### B.4 What to take over for the Device Database

Patterns that transfer directly:

- Three layers of guidance: a one-screen page next to the form (section 5), the detailed examples (sections 3 and 4), and the bot rules (section 6). Every layer links down, never up. Put a one-paragraph nutshell at the top of each.
- A short constitution of three principles, in the editorial stance already: neutral, factual, and every claim states its basis (observed, documented, reported). This replaces no original research, which would exclude the owner knowledge we want.
- Free text is for what the schema cannot hold, and anything that recurs graduates into the schema. Review periodically for candidates.
- Words to watch as cues. Reuse Wikipedia's lists (puffery, weasel words, editorialising, vagueness) as the seed for the bot's suggestion rules, and keep the no forbidden words posture.
- Checks before submission with an explanation for every hit; after-save automation only for spam and abuse. False positives get reported and fixed like bugs.
- In addition to enforcing declarative validation rules (e.g. allowed characters, structural integrity of barcodes, …), during the contribution process, the result of heuristic rules that the bot later runs for reviews should also be surfaced in the front-end to cut down on editing roundtrips
- A default that wins unless sourced: manufacturer naming for names, documentation for getting going, the most common community report for quirks.
- Guideline changes go through a visible process (an issue in the backlog repository, discussion, then the change), so the bot rules and the human guidance change together.

Patterns not to take over:

- Voting queues with waiting periods. They need a critical mass of active reviewers that the DDB does not have; the bot plus the contributor, and a trusted-contributor tier later, replaces this.
- A manual of style of Wikipedia's size. Everything that is not needed to pass the five-point check stays out of the guidance.
- Lowercase-first descriptions. Wikidata does this because descriptions are displayed as subtitles; our summary is a sentence on its own and keeps sentence case.

#### B.5 Sources for appendix B

- MusicBrainz: Style principles, Annotation, Edit and Voting pages: https://musicbrainz.org/doc/Style/Principle and https://musicbrainz.org/doc/Annotation and https://musicbrainz.org/doc/Introduction_to_Voting
- Wikipedia: Simplified ruleset, Manual of Style/Words to watch, Edit filter, ClueBot NG: https://en.wikipedia.org/wiki/Wikipedia:Simplified_ruleset and https://en.wikipedia.org/wiki/Wikipedia:Manual_of_Style/Words_to_watch and https://en.wikipedia.org/wiki/Wikipedia:Edit_filter
- Wikidata: Help:Description: https://www.wikidata.org/wiki/Help:Description
- OpenStreetMap: Good practice: https://wiki.openstreetmap.org/wiki/Good_practice

### C. Reference samples from the candidate source databases

Backlog epic OHF-Device-Database/backlog-items#27 (merge existing open databases) lists the sources we may ingest, with CSA and the blakadder Zigbee list as the first step and a provenance model (source, record id, import date per attribute) as a precondition. This section looks at those sources for one purpose only: what free text they hold, how they write it, and what that adds to our examples and rules. Matching, licensing and the import pipeline stay in the epic.

#### C.1 What each source holds as text

Reachability was tested from the drafting environment on 20 October 2026; blocked sources were read from their public repositories or from prior knowledge. The table lists the text fields each source carries and the writing convention it follows.

| Source | Text fields | Writing convention | Access on 20 Oct 2026 |
| --- | --- | --- | --- |
| Zigbee2MQTT supported devices | Description; Notes (Battery type, Pairing, Troubleshooting, device-specific control); White-label list; Exposes (structured) | Description is a noun phrase without vendor; notes are per-topic headings; pairing gives action, duration, LED state, fallback; MQTT payloads inline | Site and GitHub markdown readable |
| zigbee.blakadder.com | Page title (brand, product, model, protocol); Supports list; Pairing instructions; legacy MQTT YAML; shop links | Title crams all identifiers together; pairing text is action plus LED state; affiliate links on every page | Readable; operator approval pending |
| templates.blakadder.com (Tasmota) | GPIO table; template JSON; flashing instructions; variant notes; safety warnings | Procedural, code-heavy; safety in capitals; affiliate links | Readable |
| Z-Wave JS config DB | manufacturer (brand sold under); label (model/SKU); description (name sold under); metadata: inclusion, exclusion, reset, wakeup, manual; JSON5 comments | Written style guide: Title Case descriptions, remove restating descriptions, American English, consistency across entries | Browser is JavaScript only; config files readable on GitHub |
| OpenSmartHouse Z-Wave DB | Label; Description; Overview; Inclusion; Exclusion; Wakeup; Usage; parameter and association descriptions | Field help in the editor: short label from manufacturer model, one-line description, rich-text overview, Usage for what the manual lacks | Content is JavaScript-rendered; needs API or approval |
| ESPHome devices | Front matter (title, type, standard, board, made-for-esphome); GPIO pinout; configuration YAML; overview | Markdown per device via pull request; link generator pre-fills the file; mostly code | Readable |
| matterdatabase (GitHub JSON) | product; manufacturer (legal entity); certification_id; category; note; buyLinks | note is one free sentence, often supply-chain trivia; manufacturer is the certifying entity | JSON readable; site behind Vercel checkpoint |
| CSA certified products | Product name; company (legal entity); certification id; vendor description; compliance documents | Vendor marketing copy as description | Blocked (Cloudflare); API usage requested |
| Z-Wave Alliance products | Product name; description; manuals; inclusion/exclusion PDFs | Vendor-provided | Blocked (Cloudflare) |
| smarthome-database.eu | Structured tags only (needs no cloud, needs no hub, technology, systems, power supply); shop links; YouTube reviews | No free text | Readable |
| SmartThings, Samsung, Homey compatibility lists | Compatibility entries | Marketing lists, no reusable prose | Readable, nothing to ingest |
| domoticaeconomica.com | Unknown | Unknown | Site error on access |

#### C.2 What the sources teach

- **The one-line noun phrase is the de facto standard for a description.** Zigbee2MQTT ("Temperature and humidity sensor", "Radiator valve with thermostat"), Z-Wave JS ("Smart Door Lock", "MultiSensor 6"), OpenSmartHouse ("Short one line description of what this device is") and matterdatabase all use it. Our summary is one step richer (what is in the box, a disambiguation), so the two should probably coexist: a noun-phrase description that can be matched and translated cheaply, and a one- or two-sentence summary on top.
- **Name conventions converge with section 4.1.** Z-Wave JS: the manufacturer field carries the brand the device is sold under (Yale, not Assa Abloy); the label is the model number or SKU with the manufacturer name removed; the description is the name it is sold under, again without the manufacturer. OpenSmartHouse: "Use manufacturer defined model where known." Zigbee2MQTT separates Vendor, Model and Description and lists white-label variants (Moes HY368, Silvercrest TVR01 and others for the Tuya TS0601 valve). Two consequences: a device can carry several identifiers (Hue motion sensor is SML001 as Zigbee model and 9290012607 as retail number), which belong in identifiers, not the name; and white-label lists are a ready-made related-devices field.
- **Getting going has the same shape everywhere:** physical action, duration, what the LED or display does, and a fallback. Zigbee2MQTT adds "keep the device close to the coordinator" and a per-step description of the display for the Tuya valve. Z-Wave JS keeps four separate one-liners: inclusion, exclusion, reset, wakeup, plus a manual link; OpenSmartHouse the same four plus a free Usage field for "information not provided in the manual that may help others". Proposal: give getting going four optional sub-slots (pair, unpair, factory reset, wake) of one to three sentences each, and let the bot check that each slot names an action and a confirmation.
- **Quirks in the sources are hedged and specific.** Zigbee2MQTT on the Hue sensor: reported to have trouble re-pairing after a move from a CC2531 to a CC2652 coordinator; pairing to another network first and back again worked in one tested case. Zigbee2MQTT on Aqara: a named list of router brands through which Xiaomi devices drop off. templates.blakadder on the Sonoff S20: four plug variants, same flashing process. This is the register section 4.4 asks for, and ingested notes of this kind can go in nearly unchanged.
- **Safety warnings are a class we had not covered.** templates.blakadder: "Be sure it is NOT plugged in mains!" before a flashing step. The DDB will meet these in Tasmota and ESPHome sources and in mains-wired relays. They need a place (a quirk with a safety marker, or the first line of getting going) and a bot rule that keeps capitals and exclamation marks in a safety sentence instead of flagging them as shouting.
- **Z-Wave JS wrote our five-point check before we did.** Its style guide: descriptions that merely restate the label must be removed; units and ranges go in structured properties, not text; "only include a description if it is necessary, helpful, and adds significant value"; consistency of terminology and formatting across entries; one English variety (American). The last point is a decision we have not made: the Preview Edition and this document mix British and American spelling.
- **Guidance lives at the point of entry.** OpenSmartHouse puts a one-line help text under every field of its editor. ESPHome devices has a link generator that opens a GitHub new-file page with the front matter pre-filled. Zigbee2MQTT generates the page skeleton from the converter definition so contributors only write Notes. For the DDB contribution form this argues for the section 5 recipe as field placeholders and help texts, not as a separate page the contributor has to find.
- **Code-heavy sources confirm the link-out question.** ESPHome device pages are a GPIO table plus YAML; Zigbee2MQTT notes carry MQTT topics and payloads; templates.blakadder carries Tasmota template JSON and flashing photos. None of that is device information in the DDB sense; all of it is a reason to link to the source page with attribution. The provenance model in #27 makes that link cheap.
- **Boilerplate and commerce come with the data.** Zigbee2MQTT pages carry an auto-generated meta description ("Integrate your Aqara WSDCGQ11LM via Zigbee2MQTT with whatever smart home infrastructure you are using without the vendor's bridge or gateway"); blakadder and smarthome-database.eu carry affiliate shop links and YouTube reviews; matterdatabase carries buy links. All of it has to be stripped at ingestion, and the same detectors serve the bot for user contributions.

#### C.3 Additional examples drawn from the sources

**Name**

- Good: Hue motion sensor (SML001). Both Zigbee2MQTT identifiers (SML001, 9290012607) go in the identifiers field; the name uses the one printed on the device.
- Good: MultiSensor 6 (ZW100). Z-Wave JS label and description combined in our pattern; Aeotec as the brand sold under goes in the manufacturer field, not the name.
- Bad: Tuya TS0601_thermostat. TS0601 is a firmware model string shared by dozens of unrelated Tuya products; the sold product is Moes HY368 or Silvercrest TVR01 and so on. The model string is an identifier, the name is the product as sold.
- Bad: Aqara Temperature, Humidity and Pressure Sensor WSDCGQ11LM Zigbee compatibility. A page title from the blakadder Zigbee list; the bot splits it into manufacturer, description, model and protocol.
- Bad: Shenzhen Phaten Technology Co., Ltd. Smart Wi-Fi Plug. The CSA list and matterdatabase carry the certifying legal entity; the name takes the brand the plug is sold under, if any, and the legal entity goes in a manufacturer-of-record field.

**Summary**

- Good: Zigbee radiator valve with built-in thermostat and display, sold under several brands (Moes HY368, Silvercrest TVR01). Zigbee2MQTT's noun phrase plus the white-label fact that changes what the buyer is looking at. (*)
- Good: Z-Wave multisensor reporting motion, temperature, humidity, light, UV and vibration; USB or battery powered. Built from the Z-Wave JS entry and the manual; power options matter for placement. (*)
- Bad: Integrate your Aqara WSDCGQ11LM via Zigbee2MQTT with whatever smart home infrastructure you are using without the vendor's bridge or gateway. Generated boilerplate; identical on every Zigbee2MQTT page except the model name.
- Bad: Smart Wi-Fi Plug. The CSA product name as a summary: correct, but says nothing the category field does not. If the source gives only this, leave the summary empty rather than copying it.

**Getting going**

- Good: Press and hold the setup button on the back for about 5 seconds until the light blinks red, then press and hold again until it blinks orange; it blinks green when paired. The setup light is below the main sensor, not the indicator above it. From Zigbee2MQTT on the Hue motion sensor: action, duration, colour sequence, and the one detail people get wrong.
- Good: Include: short press the Z-Wave button on the back. Exclude: same. Factory reset: hold the button for 20 seconds, then release. From Z-Wave JS on the ZW100, as three sub-slots.
- Good: Keep the valve close to the coordinator while pairing. 1. Short press the home button to turn the display on. 2. Long press it for 3 seconds to enter settings. 3. Press plus four times until the display shows 5 and a blinking Wi-Fi logo. 4. ... From Zigbee2MQTT on the Tuya valve: each step says what the display shows, so the reader knows they are on track.
- Bad: GPIO0 Button (HIGH = off, LOW = on); GPIO12 Relay and status LED; GPIO13 Green LED, followed by a YAML configuration. ESPHome device page content: a flashing and configuration reference, which belongs behind a link, not in getting going.
- Bad: Publish to zigbee2mqtt/FRIENDLY_NAME/set with payload {"motion_sensitivity": "high"}. Integration-specific configuration; the equivalent in Home Assistant is an entity, so this is neither getting going nor a quirk.

**Quirks**

- Good: Reported to have trouble re-pairing after a move from a CC2531 to a CC2652 coordinator; pairing it to a different network first (tested with a Hue hub) and then back worked. Hedged, versioned by hardware, with a workaround.
- Good: Drops off the network when routed through routers from Centralite, GE, Iris, Ledvance, Legrand, OSRAM, Sylvania, SmartThings or Securifi; pair it directly to the coordinator or through an Aqara router. Named conditions, named fix.
- Good: Sold in four plug-type variants (EU, US, UK, AU); the flashing process is the same for each. A fact about the product line that no structured field holds yet.
- Good, safety: Never connect the serial adapter while the plug is in a mains socket. The one case where capitals in the source ("NOT plugged in mains!") are justified; the bot should not flag them.
- Borderline: Manufactured by Klite, which also makes Signify's WiZ bulbs. A matterdatabase note. Interesting supply-chain fact, but is it a quirk, a related-devices link, or trivia?
- Bad: temperature_calibration: calibrates the temperature value (absolute offset), takes effect on next report. Zigbee2MQTT option documentation; configuration reference, not a surprise.
- Bad: Available from Amazon.co.uk, Banggood, AliExpress. Shop links from blakadder; stripped at ingestion, blocked in contributions.
- Bad: YouTube: Shelly Motion, the revolutionary Wi-Fi motion sensor, first look and test. A review video link from smarthome-database.eu; not a fact about the device.

#### C.4 Consequences for ingestion and for the bot

- **Field mapping per source is mostly one-to-one:** Zigbee2MQTT Description and Z-Wave JS description to our description (or summary); Zigbee2MQTT Pairing, Z-Wave JS inclusion/exclusion/reset/wakeup and OpenSmartHouse Inclusion/Exclusion/Wakeup to getting going sub-slots; Zigbee2MQTT Troubleshooting, OpenSmartHouse Usage, matterdatabase note and blakadder variant notes to quirks; Zigbee2MQTT White-label to related devices; Exposes, Supports and GPIO tables to structured data or dropped.
- **Ingested text enters as a contribution** with the source as contributor (see appendix B.4) and is tagged with its provenance, so the bot's rules apply once, and a reader can see where a sentence came from.
- **Detectors the ingestion step needs that the bot needs too:** boilerplate sentences (identical across many entries of one source), shop and affiliate URLs, video links, code blocks (YAML, JSON, MQTT payloads), legal-entity manufacturer names, and safety sentences to preserve rather than normalise.
- **Identifiers are the matching key, not names.** Zigbee model strings, retail product numbers, Z-Wave manufacturer/product type/product id, CSA certificate ids and EANs all have to land in the identifiers field for the merge in #27 to work; the bot should ask for at least one when a contributor creates a device.

#### C.6 Sources for appendix C

- OHF-Device-Database/backlog-items#27 and the source list in its comments: https://github.com/OHF-Device-Database/backlog-items/issues/27
- Zigbee2MQTT device pages, for example WSDCGQ11LM, 9290012607 and TS0601_thermostat: https://www.zigbee2mqtt.io/supported-devices/
- Z-Wave JS config file style guide and file format: https://zwave-js.github.io/node-zwave-js/#/config-files/style-guide and https://devices.zwave-js.io/
- OpenSmartHouse Z-Wave device database (editor field help): https://opensmarthouse.org/zwavedatabase/
- ESPHome devices, adding-devices guide and Sonoff S20 page: https://devices.esphome.io/adding-devices
- blakadder Zigbee list and Tasmota templates: https://zigbee.blakadder.com/ and https://templates.blakadder.com/
- matterdatabase data and contributing guide: https://github.com/matterdatabase/matterdatabase
- smarthome-database.eu device page example: https://www.smarthome-database.eu/en/devices/shelly-motion
- Not reachable for automated access on 20 October 2026: csa-iot.org products, products.z-wavealliance.org, matterdatabase.com front end, domoticaeconomica.com, SmartThings and Homey compatibility lists (marketing lists without reusable prose).
