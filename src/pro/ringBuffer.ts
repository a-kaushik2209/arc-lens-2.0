/**
 * Fixed-capacity buffer that overwrites its oldest entry once full.
 *
 * The metric history it backs used to be a plain array with `shift()` past the
 * cap, which re-indexes every remaining element on every step — an O(n) cost
 * paid in the extension host for the entire remainder of a long run. Writing
 * into a rotating slot keeps insertion O(1) no matter how many steps arrive.
 */
export class RingBuffer<T> {
  private items: T[] = [];
  private next = 0;

  constructor(private readonly capacity: number) {
    if (!Number.isInteger(capacity) || capacity < 1) {
      throw new RangeError(`RingBuffer capacity must be a positive integer, got ${capacity}`);
    }
  }

  push(item: T): void {
    if (this.items.length < this.capacity) {
      this.items.push(item);
      return;
    }
    this.items[this.next] = item;
    this.next = (this.next + 1) % this.capacity;
  }

  /** Entries in insertion order, oldest first. */
  toArray(): T[] {
    if (this.items.length < this.capacity) return this.items.slice();
    return this.items.slice(this.next).concat(this.items.slice(0, this.next));
  }

  get length(): number {
    return this.items.length;
  }

  clear(): void {
    this.items = [];
    this.next = 0;
  }
}
