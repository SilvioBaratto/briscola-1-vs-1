import { Component, input, output, ChangeDetectionStrategy } from '@angular/core';
import { CardComponent } from '../card/card.component';
import type { HandCard } from '../../models/game.models';

@Component({
  selector: 'app-hand',
  imports: [CardComponent],
  templateUrl: './hand.component.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class HandComponent {
  readonly cards = input.required<HandCard[]>();
  readonly isPlayerTurn = input(false);
  readonly selectedCardIndex = input<number | null>(null);

  readonly cardSelected = output<number>();

  handleCardClick(index: number): void {
    if (this.isPlayerTurn()) {
      this.cardSelected.emit(index);
    }
  }
}
