import { Component, input, output, ChangeDetectionStrategy } from '@angular/core';
import type { ModelStatus } from '../../models/game.models';

@Component({
  selector: 'app-training-panel',
  imports: [],
  templateUrl: './training-panel.component.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class TrainingPanelComponent {
  readonly modelStatus = input<ModelStatus | null>(null);
  readonly isLoading = input(false);
  readonly error = input<string | null>(null);

  readonly trainClick = output<void>();
  readonly newGameClick = output<void>();
  readonly saveModelClick = output<void>();

  readonly Math = Math;

  readonly bufferSize = (): number => {
    return this.modelStatus()?.buffer_size ?? 0;
  };
}
