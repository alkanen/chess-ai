import type uPlot from 'uplot';

/**
 * Stands in for uPlot, which draws on a canvas jsdom does not have. It keeps what a chart
 * was made with and the data it was last given, which is what the tests look at.
 */
export class FakeUPlot {
  static instances: FakeUPlot[] = [];

  data: uPlot.AlignedData;
  /** Whether the last data it was given rescaled the axes to fit, as an unzoomed chart does. */
  rescaled = true;
  destroyed = false;
  readonly over = document.createElement('div');

  constructor(
    readonly options: uPlot.Options,
    data: uPlot.AlignedData,
    readonly target: HTMLElement,
  ) {
    this.data = data;
    FakeUPlot.instances.push(this);
  }

  /** The charts on show now, by the label of their first data series. */
  static live(): FakeUPlot[] {
    return FakeUPlot.instances.filter((chart) => !chart.destroyed);
  }

  /** The chart on show whose series include one called `label`. */
  static withSeries(label: string): FakeUPlot {
    const chart = FakeUPlot.live().find((candidate) =>
      candidate.options.series.some((series) => series.label === label),
    );
    if (chart === undefined) {
      throw new Error(`no chart has a series called ${label}`);
    }
    return chart;
  }

  setData(data: uPlot.AlignedData, resetScales = true) {
    this.data = data;
    this.rescaled = resetScales;
  }

  /** The viewer drags across the chart to zoom in, as uPlot reports it. */
  select() {
    const hooks = this.options.hooks?.setSelect;
    (Array.isArray(hooks) ? hooks : [hooks]).forEach((hook) => hook?.(this as unknown as uPlot));
  }

  setSize() {}

  destroy() {
    this.destroyed = true;
  }
}
