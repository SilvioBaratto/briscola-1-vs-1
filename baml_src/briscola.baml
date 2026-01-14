// Briscola card game related functions

// Simple function to test if the LLM knows about Briscola
function TestBriscolaKnowledge(question: string) -> string {
  client Model
  prompt #"
    {{ _.role("user") }}
    {{ question }}

    Answer in English.
  "#
}

// Test case for Briscola knowledge
test briscola_rules {
  functions [TestBriscolaKnowledge]
  args {
    question "Do you know the rules of Briscola?"
  }
}
