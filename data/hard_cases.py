"""Curated hard-case templates for turn detection.

Each entry is (template, label) where label is 1 = COMPLETE turn, 0 = INCOMPLETE.
Templates may contain placeholders expanded by build_dataset.py:
  {d}    -> a random digit word ("five") or digit ("5")
  {name} -> a random first name
  {city} -> a random city
  {item} -> a random noun phrase object
"""

# ---------------------------------------------------------------- COMPLETE ---
COMPLETE = [
    # short answers / backchannels that ARE full turns
    "Yeah.", "Yes.", "No.", "Okay.", "Sure.", "That's right.", "Sounds good.",
    "No thanks.", "Yes please.", "I think so.", "I don't think so.", "Correct.",
    "Exactly.", "Not really.", "Maybe later.", "That works for me.",
    "Done.", "All set.", "Nothing else, thanks.", "That's all.",
    "That would be great.", "I'm not sure.", "I have no idea.",
    # complete questions
    "What time do you close?", "Can I change my reservation?",
    "How much does that cost?", "Do you deliver to {city}?",
    "Is there anything cheaper?", "What's your name?",
    "Could you repeat that?", "Can you hear me?",
    # complete statements with entities/numbers (fully given)
    "My phone number is {d}{d}{d} {d}{d}{d} {d}{d}{d}{d}.",
    "My zip code is {d}{d}{d}{d}{d}.",
    "The confirmation number is {d}{d}{d}{d}{d}{d}.",
    "My name is {name}.",
    "It's {name}, spelled {spelling}.",
    "I live in {city}.",
    "I'd like to book a table for two at seven pm.",
    "I want to cancel my subscription.",
    "I'm calling about my order from last week.",
    "Please ship it to my home address.",
    "I'll take the second option.",
    "Let's go with the earlier flight.",
    "My email is {name} at gmail dot com.",
    "I was born on the {d}th of May, nineteen ninety {d}.",
    "There are {d} people in my party.",
    "I need it by Friday.",
    "Actually, never mind.",
    "That's everything I needed, thank you.",
    "Goodbye.", "Thanks, bye.",
    # complete despite starting with a filler
    "Um, yes.", "Uh, that works.", "Well, okay then.",
    "Hmm, I'll take it.", "Oh, I see.",
]

# -------------------------------------------------------------- INCOMPLETE ---
INCOMPLETE = [
    # bare fillers / hesitations
    "Um", "Uh", "Hmm", "Well", "So", "Let me think", "Let me see",
    "Give me a second", "Hold on", "Uh, let me check", "One moment, it's",
    "Um, so", "Okay so", "Right, so basically",
    # trailing conjunctions / prepositions / articles
    "I want to cancel my", "I was thinking that maybe we could",
    "Can you send it to", "I'd like to order the", "My address is",
    "The problem is that", "I called yesterday because", "It's about the",
    "I need to change my flight from", "We could either do that or",
    "If it doesn't arrive by Friday then", "I'd prefer the one with",
    "The reason I'm calling is", "So what happened was",
    "Could you tell me the", "I'm looking for a", "Do you have any",
    "My question is about the", "Before we finish, can I also",
    "And also the", "But the thing is", "Because I already",
    # mid-number / mid-entity / mid-spelling
    "My phone number is {d}{d}{d}", "My phone number is {d}{d}{d} {d}{d}{d}",
    "My zip code is {d}{d}", "The confirmation number is {d}{d}{d}{d}",
    "It's spelled {partial_spelling}", "My email is {name} at",
    "My email is {name} dot", "I live at {d}{d} Elm",
    "My card ends in {d}{d}", "The flight number is BA {d}",
    "I was born on the {d}th of",
    "It's {name}, spelled",
    "My last name is {name}, that's {partial_spelling}",
    # mid-list
    "I need three things: first,", "We have two options. One,",
    "I'd like a burger, fries, and", "Monday, Wednesday, and",
    # restarts / self-corrections in progress
    "I want the... actually make it the", "No wait, I meant the",
    "Sorry, what I meant to say is", "Let me rephrase that, I",
]
