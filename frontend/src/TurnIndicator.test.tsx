import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { TurnIndicator } from './TurnIndicator';

describe('TurnIndicator', () => {
  it('shows that White is to move', () => {
    render(<TurnIndicator turn="white" />);

    expect(screen.getByRole('status')).toHaveTextContent('White to move');
  });

  it('shows that Black is to move', () => {
    render(<TurnIndicator turn="black" />);

    expect(screen.getByRole('status')).toHaveTextContent('Black to move');
  });
});
