/**
 * The addresses of the datasets page and of a dataset's games, so that a reload, a link and the
 * browser's Back button all come back to the same page of the same split.
 *
 * Pages and games are numbered from 1 in an address, as they are on the page, and from 0
 * everywhere else.
 */

/** How many games one page of a dataset lists. */
export const PAGE_SIZE = 50;

/** The split a dataset's games are listed from until another is chosen. */
export const DEFAULT_SPLIT = 'train';

/** One page of one split of a dataset. */
export interface DatasetPage {
  name: string;
  split: string;
  /** Counted from 0. */
  page: number;
}

/** One game of one split of a dataset. */
export interface DatasetGameRef {
  name: string;
  split: string;
  /** Counted from 0, as the server counts it. */
  index: number;
}

const DATASET_PAGE = /^#datasets\/([^/]+)(?:\/([^/]+)\/(\d+))?$/;
const DATASET_GAME = /^#replay\/dataset\/([^/]+)\/([^/]+)\/(\d+)$/;

/** The address of a page of a dataset's games, as short as it can be for the first one. */
export function datasetHash({ name, split, page }: DatasetPage): string {
  const dataset = `#datasets/${encodeURIComponent(name)}`;
  if (split === DEFAULT_SPLIT && page === 0) {
    return dataset;
  }
  return `${dataset}/${encodeURIComponent(split)}/${page + 1}`;
}

/** The page of the dataset an address names, or null for any other address. */
export function datasetInHash(hash: string): DatasetPage | null {
  const found = DATASET_PAGE.exec(hash);
  if (found === null) {
    return null;
  }
  return {
    name: decodeURIComponent(found[1]),
    split: found[2] === undefined ? DEFAULT_SPLIT : decodeURIComponent(found[2]),
    page: found[3] === undefined ? 0 : Math.max(Number(found[3]) - 1, 0),
  };
}

/** The address of one game of a dataset in the replay viewer. */
export function datasetGameHash({ name, split, index }: DatasetGameRef): string {
  return `#replay/dataset/${encodeURIComponent(name)}/${encodeURIComponent(split)}/${index + 1}`;
}

/** The dataset game an address opens in the replay viewer, or null for any other address. */
export function datasetGameInHash(hash: string): DatasetGameRef | null {
  const found = DATASET_GAME.exec(hash);
  if (found === null || Number(found[3]) < 1) {
    return null;
  }
  return {
    name: decodeURIComponent(found[1]),
    split: decodeURIComponent(found[2]),
    index: Number(found[3]) - 1,
  };
}

/** The page of its dataset a game is listed on. */
export function pageOf({ name, split, index }: DatasetGameRef): DatasetPage {
  return { name, split, page: Math.floor(index / PAGE_SIZE) };
}
