// Undo and redo over sources: each visual edit is one step, a burst of typing another.

const BURST_MS = 1000;
const LIMIT = 500;

export class History {
  constructor(source) {
    this.entries = [source];
    this.at = 0;
    this.typing = false;
    this.last = 0;
  }

  get current() {
    return this.entries[this.at];
  }

  /** Record a source; typing within a second of the last keystroke joins its step. */
  push(source, { typing = false } = {}) {
    if (source === this.current) return;
    const now = performance.now();
    this.entries.length = this.at + 1;
    if (typing && this.typing && now - this.last < BURST_MS && this.at > 0) {
      this.entries[this.at] = source;
    } else {
      this.entries.push(source);
      this.at += 1;
    }
    this.typing = typing;
    this.last = now;
    if (this.entries.length > LIMIT) (this.entries.shift(), (this.at -= 1));
  }

  /** The source before, or null at the start. */
  undo() {
    this.typing = false;
    return this.at > 0 ? this.entries[--this.at] : null;
  }

  /** The source after, or null at the end. */
  redo() {
    this.typing = false;
    return this.at < this.entries.length - 1 ? this.entries[++this.at] : null;
  }
}
