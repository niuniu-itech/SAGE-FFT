# Hosted decoder controls

Run schema_flat and schema_typed on M3 with seeds 41, 42 and 43 and 12 proposals
per trial. Both request a diagnosis string and a decision in a strict JSON
schema, use temperature 0.4 and allow 512 output tokens. Candidate menus,
history and compiler checks are shared. The two interfaces differ in their
decision representation. Independent model calls may overlap; target timings
are serialized. Invalid answers remain in the proposal denominator.
The combined hosted study and schema matrix contains 252 proposals.
