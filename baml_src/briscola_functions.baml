// Briscola LLM opponent - optimized for Ollama speed

// Main function: LLM chooses which card to play
function ChooseCard(
  hand: string[],
  briscola: string,
  history: string,
  opponent_card: string?
) -> Action {
  client Model
  prompt #"
    {{ _.role("user") }}
    === STATE ===
    Trump: {{ briscola }}
    {% if opponent_card -%}
    Opp: {{ opponent_card }}
    {% endif %}
    {% if history -%}
    Played:
    {{ history }}
    {% endif %}
    === HAND ===
    {% for card in hand -%}
    {{ loop.index0 }}:{{ card }} {% endfor %}

    Pick: {% if hand|length == 1 %}0{% elif hand|length == 2 %}0,1{% else %}0,1,2{% endif %}

    {{ ctx.output_format }}
  "#
}

// Test cases
test test_choose_card_leading {
  functions [ChooseCard]
  args {
    hand ["A♦", "3♠", "K♥"]
    briscola "coins"
    history "- ♥: 7,J\n- ♠: 4\n- ♣: 2,4"
    opponent_card null
  }
}

test test_choose_card_responding {
  functions [ChooseCard]
  args {
    hand ["7♦", "J♠", "5♥"]
    briscola "swords"
    history "- ♦: A,3\n- ♥: 2\n- ♣: K"
    opponent_card "K♥"
  }
}