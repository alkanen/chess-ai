import type { ReplayFile } from '../api';
import files from './fixtures/replay-two-games.json';

/**
 * A PGN file of two games as the server reads it back, once per game that can be the
 * one selected: fool's mate, then a scholar's mate. A pytest test keeps the fixture in
 * step with the server.
 */
export const [foolsMateFile, scholarsMateFile] = files as unknown as [ReplayFile, ReplayFile];
