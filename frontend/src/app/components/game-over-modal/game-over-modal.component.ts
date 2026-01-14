import { Component, input, output, ChangeDetectionStrategy } from '@angular/core';

@Component({
  selector: 'app-game-over-modal',
  imports: [],
  templateUrl: './game-over-modal.component.html',
  styleUrl: './game-over-modal.component.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class GameOverModalComponent {
  readonly isVisible = input(false);
  readonly winner = input<string | null>(null);
  readonly playerScore = input(0);
  readonly opponentScore = input(0);
  readonly message = input<string | null>(null);

  readonly playAgainClick = output<void>();
  readonly closeClick = output<void>();

  handleBackdropClick(event: MouseEvent): void {
    this.closeClick.emit();
  }
}
