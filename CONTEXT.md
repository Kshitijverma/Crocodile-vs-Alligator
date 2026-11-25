# Crocodilian Classifier

Tells a visitor which group of crocodilian is in a photo they upload: alligator, caiman, crocodile or gharial.

## Language

### Animals

**Crocodilian**:
Any member of the order Crocodylia. The four Groups below cover every living crocodilian.
_Avoid_: Reptile (too broad), croc (ambiguous with Crocodile)

**Group**:
One of the four answers the classifier can give. Each Group is defined by taxonomy, not by appearance.
_Avoid_: Class, category, species

**Alligator**:
A member of the genus *Alligator*: the American alligator and the Chinese alligator.
_Avoid_: Gator

**Caiman**:
A member of the genera *Caiman*, *Melanosuchus* (black caiman) or *Paleosuchus* (dwarf caimans).

**Crocodile**:
A true crocodile: a member of the genera *Crocodylus*, *Mecistops* (slender-snouted crocodiles) or *Osteolaemus* (dwarf crocodiles).
_Avoid_: Croc

**Gharial**:
A member of the genera *Gavialis* (gharial) or *Tomistoma* (false gharial).
_Avoid_: Gavial

### Answers

**Verdict**:
What the site tells the visitor about one photo: either a Group, or Can't tell.
_Avoid_: Prediction, result

**Confidence**:
The model's probability for a Group on one photo, from 0 to 1. Every Verdict shows the Confidence for all four Groups.
_Avoid_: Score, certainty

**Can't tell**:
The Verdict given when no Group's Confidence reaches the Threshold. Expected for photos that are not of a Crocodilian, or are too unclear to judge.
_Avoid_: Unknown, other, none

**Threshold**:
The lowest Confidence at which the site names a Group instead of saying Can't tell.

### Data

**Observation**:
One sighting record on iNaturalist, with one or more photos and a community-verified identification. The unit used to keep training and test photos apart.

**Credit**:
The attribution owed for one training photo: its observer, licence and source link.
