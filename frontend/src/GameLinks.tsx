import { useMemo, useState } from 'react';
import type { Access, Color } from './api';
import { gameAddress, rememberedGame } from './myGames';
import './GameLinks.css';

const SIDES: Record<Color, string> = { white: 'White', black: 'Black' };

/** One link to the game, written out to be copied and sent. */
function ShareLink({ label, note, address }: { label: string; note: string; address: string }) {
  const [copied, setCopied] = useState(false);

  async function copy() {
    try {
      await navigator.clipboard.writeText(address);
      setCopied(true);
    } catch {
      // A browser that will not let the page write to the clipboard, such as one on a page
      // served over plain HTTP: the address is there to be selected and copied by hand.
      setCopied(false);
    }
  }

  return (
    <div className="share-link">
      <label>
        {label}
        <input type="text" readOnly value={address} onFocus={(e) => e.target.select()} />
      </label>
      <button type="button" onClick={() => void copy()}>
        {copied ? 'Copied' : 'Copy'}
      </button>
      <p className="note">{note}</p>
    </div>
  );
}

interface GameLinksProps {
  /** The link this page was opened through. */
  link: string;
  access: Access;
  /** The game's watch link. */
  watch: string;
}

/**
 * The links to the game, to keep and to pass on. A link is the only way back to a game, and
 * whoever holds one can do what it allows, so each says what that is.
 *
 * The other player's link is only here in the browser that started the game, which was given
 * it to send to them: the server hands a play link to nobody else.
 */
export function GameLinks({ link, access, watch }: GameLinksProps) {
  // Read once rather than on every move: what this browser was given does not change.
  const started = useMemo(() => rememberedGame(link)?.links, [link]);
  const other: Color | null = access === 'white' ? 'black' : access === 'black' ? 'white' : null;
  const theirs = other !== null ? (started?.[other] ?? null) : null;

  return (
    <section className="game-links" aria-labelledby="game-links-heading">
      <h2 id="game-links-heading">Links</h2>
      {access === 'white' || access === 'black' ? (
        <ShareLink
          label={`Your link, playing ${SIDES[access]}`}
          note="Keep it: it is the only way back to this game, and it plays your side."
          address={gameAddress(link, access)}
        />
      ) : access === 'control' ? (
        <ShareLink
          label="Your link"
          note="Keep it: it is the only link that can abort this game."
          address={gameAddress(link, access)}
        />
      ) : null}
      {other !== null && theirs !== null && (
        <ShareLink
          label={`${SIDES[other]}'s link`}
          note={`Send this to whoever plays ${SIDES[other]}: it plays their side, and only theirs.`}
          address={gameAddress(theirs, other)}
        />
      )}
      <ShareLink
        label="Watch link"
        note="For anyone who wants to follow the game. It can do nothing to it."
        address={gameAddress(watch, 'watch')}
      />
    </section>
  );
}
