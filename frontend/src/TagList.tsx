import './TagList.css';

interface TagListProps {
  tags: string[];
  /** Called with a tag when it is clicked; without it the tags are only shown. */
  onPick?: (tag: string) => void;
  /** The tag being filtered by, if any, which is shown as pressed. */
  picked?: string | null;
}

/** A run's tags, as chips. */
export function TagList({ tags, onPick, picked = null }: TagListProps) {
  return (
    <ul className="tag-list" aria-label="Tags">
      {tags.map((tag) => (
        <li key={tag}>
          {onPick === undefined ? (
            <span className="tag">{tag}</span>
          ) : (
            <button
              type="button"
              className="tag"
              aria-pressed={picked === tag}
              title={`Show only the runs tagged ${tag}`}
              onClick={() => onPick(tag)}
            >
              {tag}
            </button>
          )}
        </li>
      ))}
    </ul>
  );
}
