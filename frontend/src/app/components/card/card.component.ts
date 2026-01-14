import { Component, input, output, computed, ChangeDetectionStrategy } from '@angular/core';
import { NgOptimizedImage } from '@angular/common';
import type { CardSchema } from '../../models/game.models';
import { getCardImagePath, getCardBackImagePath } from '../../utils/card-mapper';

@Component({
  selector: 'app-card',
  imports: [NgOptimizedImage],
  templateUrl: './card.component.html',
  styleUrl: './card.component.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class CardComponent {
  readonly card = input<CardSchema | null>(null);
  readonly isHidden = input(false);
  readonly isClickable = input(false);
  readonly isSelected = input(false);
  readonly showPoints = input(true);
  readonly size = input<'small' | 'medium' | 'large'>('medium');

  readonly cardClick = output<void>();

  readonly cardBackPath = getCardBackImagePath();

  readonly imagePath = computed(() => {
    const cardData = this.card();
    if (!cardData) return this.cardBackPath;
    return getCardImagePath(cardData.value, cardData.suit);
  });

  readonly cardClasses = computed(() => {
    const sizeClass = this.getSizeClass();
    return {
      [sizeClass]: true,
      'hover:scale-105 hover:-translate-y-2': this.isClickable(),
      'opacity-50 cursor-not-allowed': !this.isClickable() && !this.isHidden(),
      'scale-105 -translate-y-2': this.isSelected(),
    };
  });

  private getSizeClass(): string {
    switch (this.size()) {
      case 'small':
        return 'w-14 h-[5.25rem] sm:w-16 sm:h-24';
      case 'large':
        return 'w-20 h-[7.5rem] sm:w-24 sm:h-36 md:w-28 md:h-[10.5rem]';
      default:
        return 'w-16 h-24 sm:w-20 sm:h-[7.5rem] md:w-24 md:h-36';
    }
  }

  handleClick(): void {
    if (this.isClickable()) {
      this.cardClick.emit();
    }
  }
}
