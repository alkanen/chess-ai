import { useEffect, useId, useRef, useState, type FormEvent } from 'react';
import { saveRunNotes, type RunNotes } from './api';
import { TagList } from './TagList';
import './RunNotesPanel.css';

const NO_NOTES: RunNotes = { title: null, tags: [], notes: '' };

function sameNotes(a: RunNotes | null, b: RunNotes | null): boolean {
  if (a === null || b === null) {
    return a === b;
  }
  return (
    a.title === b.title &&
    a.notes === b.notes &&
    a.tags.length === b.tags.length &&
    a.tags.every((tag, index) => tag === b.tags[index])
  );
}

/** Notes saved here that the run channel has not caught up with yet. */
interface Pending {
  /** What the last save kept, which is what is shown. */
  notes: RunNotes;
  /** What the channel may still send that is older than that: what the first of the saves
   * replaced, and what every save before the last one kept. */
  older: (RunNotes | null)[];
}

/**
 * A run's notes as they are to be shown: the ones just saved here, until the run channel
 * brings notes newer than those.
 *
 * The channel cannot simply be believed from the save on. It reads the notes once a second
 * and sends them whenever anything about the run changed, so a message read just before a
 * save can arrive just after it, carrying older notes — sent because the heartbeat moved.
 * Older means what the saves replaced, including the earlier of two saves made within a
 * second. Anything else was read after the last save: either what it kept, or what somebody
 * wrote since, from another tab or the command line, which then wins.
 */
export function useShownNotes(
  followed: RunNotes | null,
): [RunNotes | null, (saved: RunNotes) => void] {
  const [pending, setPending] = useState<Pending | null>(null);
  const current = useRef(followed);
  current.current = followed;
  useEffect(() => {
    if (pending !== null && !pending.older.some((older) => sameNotes(followed, older))) {
      setPending(null);
    }
  }, [followed, pending]);
  return [
    pending?.notes ?? followed,
    (notes: RunNotes) =>
      setPending((before) => ({
        notes,
        older: before === null ? [current.current] : [...before.older, before.notes],
      })),
  ];
}

/** Tags as a person types them: separated by spaces or commas, or both. */
export function parseTags(text: string): string[] {
  return text.split(/[\s,]+/).filter((tag) => tag !== '');
}

interface RunNotesPanelProps {
  run: string;
  /** What the run has, or null while it cannot be read; editing then starts from nothing. */
  notes: RunNotes | null;
  onSaved: (notes: RunNotes) => void;
}

/** A run's tags and notes, and a form to change them along with its title. */
export function RunNotesPanel({ run, notes, onSaved }: RunNotesPanelProps) {
  const [draft, setDraft] = useState<{ title: string; tags: string; notes: string } | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const id = useId();
  const shown = notes ?? NO_NOTES;

  function edit() {
    setDraft({ title: shown.title ?? '', tags: shown.tags.join(' '), notes: shown.notes });
    setError(null);
  }

  async function save(event: FormEvent) {
    event.preventDefault();
    if (draft === null) {
      return;
    }
    setSaving(true);
    setError(null);
    try {
      const kept = await saveRunNotes(run, {
        title: draft.title.trim() === '' ? null : draft.title,
        tags: parseTags(draft.tags),
        notes: draft.notes,
      });
      onSaved(kept);
      setDraft(null);
    } catch (e) {
      // The form stays as it was typed, so that nothing is lost to a refusal.
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  }

  if (draft === null) {
    return (
      <section className="run-notes" aria-label="Notes">
        {shown.tags.length > 0 && <TagList tags={shown.tags} />}
        {shown.notes.trim() === '' ? (
          <p className="note">No notes.</p>
        ) : (
          <p className="text">{shown.notes}</p>
        )}
        <button type="button" onClick={edit}>
          Edit title, tags and notes
        </button>
      </section>
    );
  }

  return (
    <form className="run-notes editing" aria-label="Edit notes" onSubmit={save}>
      <label htmlFor={`${id}-title`}>Title</label>
      <input
        id={`${id}-title`}
        value={draft.title}
        placeholder={run}
        maxLength={200}
        onChange={(e) => setDraft({ ...draft, title: e.target.value })}
      />
      <label htmlFor={`${id}-tags`}>Tags</label>
      <input
        id={`${id}-tags`}
        value={draft.tags}
        placeholder="separated by spaces or commas"
        aria-describedby={`${id}-tags-hint`}
        onChange={(e) => setDraft({ ...draft, tags: e.target.value })}
      />
      <span id={`${id}-tags-hint`} className="hint">
        One word each, such as mlp or wide.
      </span>
      <label htmlFor={`${id}-notes`}>Notes</label>
      <textarea
        id={`${id}-notes`}
        value={draft.notes}
        rows={6}
        onChange={(e) => setDraft({ ...draft, notes: e.target.value })}
      />
      {error !== null && (
        <p role="alert" className="error">
          Could not save: {error}
        </p>
      )}
      <div className="actions">
        <button type="submit" disabled={saving}>
          {saving ? 'Saving…' : 'Save'}
        </button>
        <button type="button" disabled={saving} onClick={() => setDraft(null)}>
          Cancel
        </button>
      </div>
    </form>
  );
}
