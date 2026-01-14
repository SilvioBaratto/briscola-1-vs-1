/**
 * Maps card values to image file numbers
 * Cards are named: {suit}{number}.png where:
 * - Suits: bastoni, coppe, denara, spade
 * - Numbers 1-10 map to card values
 */

const CARD_VALUE_TO_NUMBER: Record<string, number> = {
  ACE: 1,
  TWO: 2,
  THREE: 3,
  FOUR: 4,
  FIVE: 5,
  SIX: 6,
  SEVEN: 7,
  JACK: 8,
  HORSE: 9,
  KING: 10,
};

const SUIT_TO_IMAGE_NAME: Record<string, string> = {
  clubs: 'bastoni',
  cups: 'coppe',
  coins: 'denara',
  swords: 'spade',
};

export function getCardImagePath(value: string, suit: string): string {
  const number = CARD_VALUE_TO_NUMBER[value];
  const suitName = SUIT_TO_IMAGE_NAME[suit.toLowerCase()];

  if (!number || !suitName) {
    console.error('Invalid card:', { value, suit });
    return '/assets/cards/card-back.png';
  }

  return `/assets/cards/${suitName}${number}.png`;
}

export function getCardBackImagePath(): string {
  return '/assets/cards/card-back.png';
}
