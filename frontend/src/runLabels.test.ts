import { describe, expect, it } from 'vitest';
import type { RunSummary } from './api';
import { describeRun, runLabels } from './runLabels';

const LONG =
  '2026-10-01T00h05m_mlp-lichess_b8k_lr1e-3_epochs4_depth6_width4096_mlrf0.1_wu3000_vw0.5_side-to-move';

function run(name: string, changes: Partial<RunSummary> = {}): RunSummary {
  return {
    name,
    title: null,
    tags: [],
    architecture: 'mlp',
    created: '2026-10-01T00:05:00Z',
    status: 'finished',
    step: 344_000,
    steps: 344_000,
    checkpoints: 5,
    ...changes,
  };
}

function labels(...runs: RunSummary[]): string[] {
  const named = runLabels(runs);
  return runs.map((summary) => named.get(summary.name) ?? '?');
}

describe('runLabels', () => {
  it('shows a run by its title when it has one', () => {
    expect(labels(run(LONG, { title: 'mlp6x4096' }))).toEqual(['mlp6x4096']);
  });

  it('shows a short name without the time it starts with', () => {
    expect(labels(run('2026-10-02T19h17m_resnet10x128'), run('mlp-baseline'))).toEqual([
      'resnet10x128',
      'mlp-baseline',
    ]);
  });

  it('cuts a long name down to which experiment it is', () => {
    expect(labels(run(LONG))).toEqual(['mlp-lichess…']);
  });

  it('tells apart runs whose long names differ only at the end by when they started', () => {
    expect(
      labels(run(`${LONG}_history1`), run(LONG.replace('00h05m', '07h30m') + '_history2')),
    ).toEqual(['mlp-lichess… (2026-10-01T00h05m)', 'mlp-lichess… (2026-10-01T07h30m)']);
  });

  it('says when each of two runs with the same title started', () => {
    expect(
      labels(
        run(LONG, { title: 'mlp6x4096' }),
        run(`2026-10-01T07h30m_mlp-lichess_b8k_history2`, { title: 'mlp6x4096' }),
        run('2026-10-02T19h17m_resnet10x128', { title: 'resnet10x128' }),
      ),
    ).toEqual(['mlp6x4096 (2026-10-01T00h05m)', 'mlp6x4096 (2026-10-01T07h30m)', 'resnet10x128']);
  });

  it('falls back on the whole name when cutting two names down makes them the same', () => {
    const other = `${LONG}_history2`;

    expect(labels(run(LONG), run(other))).toEqual([LONG, other]);
  });
});

describe('describeRun', () => {
  it('says how far a running run has got and how many checkpoints it has, briefly', () => {
    const summary = run('r', { status: 'running', step: 215_272, steps: 344_000 });

    expect(describeRun(summary, 'resnet10x128')).toBe(
      'resnet10x128 · 215k/344k steps, 5 checkpoints',
    );
  });

  it.each([
    [11_600, 12_000, '11k/12k'],
    [999_600, 1_000_000, '999k/1.00M'],
    // Rounded alike however it is done, so said in full.
    [1_003_000, 1_004_000, '1,003,000/1,004,000'],
  ])('never makes a run at step %i of %i look finished', (step, steps, progress) => {
    const summary = run('r', { status: 'running', step, steps, checkpoints: 2 });

    expect(describeRun(summary, 'r')).toBe(`r · ${progress} steps, 2 checkpoints`);
  });

  it('gives a finished run its length, and counts one checkpoint as one', () => {
    expect(describeRun(run('r', { checkpoints: 1, step: 20_000, steps: 20_000 }), 'sweep')).toBe(
      'sweep · 20k steps, 1 checkpoint',
    );
  });

  it('leaves out the steps of a run that does not say', () => {
    expect(describeRun(run('r', { step: null, steps: null }), 'old')).toBe('old · 5 checkpoints');
  });
});
