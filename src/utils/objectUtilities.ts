/**
 * Create an object from an array of keys and an array of objects
 * @param keys
 * @param values
 */
export const keysValuesToObject = <K extends PropertyKey, V>(keys: K[], values: V[]): Record<K, V> => {
    return Object.fromEntries(keys.map((key, i) => [key, values[i]])) as Record<K, V>;
}

/**
 * Allows key-value objects to be mapped similarly to array.map.
 * Preserves the keys but maps the values.
 * @param object
 * @param mapFunction
 */
const mapObject = <K extends string, V, NV>(object: Record<K, V>, mapFunction: (value: V, key: K, index: number, object: Record<K, V>) => NV): Record<K, NV> => {
    const keys = Object.keys(object) as K[];
    const newValues = Object.values<V>(object).map((value, index) => mapFunction(value, keys[index], index, object));
    return keysValuesToObject(keys, newValues);
}

/**
 * Create an object by mapping keys to values.
 * @param keys
 * @param createValue
 */
const objectFromKeys = <K extends string, V>(keys: K[], createValue: (key: K, index: number, keyArray: K[]) => V): Record<K, V> => {
    const values = keys.map(createValue);
    return keysValuesToObject(keys, values);
}

/**
 * Create an object from an array of values by mapping each value to its key
 * @param values
 * @param createKey
 */
const objectFromValues = <K extends string, V>(values: V[], createKey: (value: V, index: number, valueArray: V[]) => K): Record<K, V> => {
    const keys = values.map(createKey);
    return keysValuesToObject(keys, values);
}

export class ObjectBuilder<K extends PropertyKey = string, V = any> {
    private constructor(private _keys: K[], private _values: V[]) {
    }
    get object() {
        return keysValuesToObject(this._keys, this._values);
    }

    /**
     * Use the existing values.
     * Replace the entire set of keys with a new array.
     */
    replaceKeys<NK extends PropertyKey>(keys: NK[]): Record<NK, V> {
        return keysValuesToObject(keys, this._values);
    }

    /**
     * Use the existing keys.
     * Replace the entire set of values with a new array.
     */
    replaceValues<NV>(values: NV[]): Record<K, NV> {
        return keysValuesToObject(this._keys, values);
    }

    /**
     * Use the existing keys.
     * Set new values based on the current value, key, index, and object.
     *
     * Allows key-value objects to be mapped similarly to array.map.
     */
    mapValues<NV>( mapFunction: (value: V, key: K, index: number, object: Record<K, V>) => NV): Record<K, NV> {
        return this.replaceValues(
            this._keys.map((key, index) => mapFunction(this._values[index], key, index, this.object))
        )
    }

    /**
     * Use the existing values.
     * Set new keys based on the current key, value, index, and object.
     */
    mapKeys<NK extends PropertyKey>( mapFunction: (key: K, value: V, index: number, object: Record<K, V>) => NK): Record<NK, V> {
        return this.replaceKeys(
            this._keys.map((key, index) => mapFunction(key, this._values[index], index, this.object))
        )
    }

    /**
     * Use the existing values.
     * Set new keys based on the values.
     */
    createKeys<Key extends string>(keyCreator: (value: V, index: number, valueArray: V[]) => Key): Record<Key, V> {
        return this.replaceKeys(this._values.map(keyCreator));
    }

    /**
     * Use the existing keys.
     * Set new values based on the keys.
     */
    createValues<Value>(valueCreator: (key: K, index: number, keyArray: K[]) => Value): Record<K, Value> {
        return this.replaceValues(this._keys.map(valueCreator));
    }

    /**
     * Create an ObjectBuilder instance from an existing object or from an array.
     * An array of strings or numbers can be used as either keys or values, but other types can only be values.
     * Note: objects with number keys cannot be mapped properly because Object.keys() returns strings.
     */
    public static from<T extends PropertyKey>(arrayOrObject: T[]): Pick<ObjectBuilder<T, T>, 'createKeys' | 'createValues'>;
    public static from<T>(arrayOrObject: T[]): Pick<ObjectBuilder<string, T>, 'createKeys'>;
    public static from<TK extends string, TV>(arrayOrObject: Record<TK, TV>): ObjectBuilder<TK, TV>;
    public static from(arrayOrObject: any[] | Record<any, any>): any {
        if ( Array.isArray(arrayOrObject) ) {
            // strings can be keys or values, but other types can only be values.
            const keys = arrayOrObject.every(v => typeof v === "string") ? arrayOrObject : [];
            return new ObjectBuilder(keys, arrayOrObject);
        } else {
            return new ObjectBuilder(Object.keys(arrayOrObject), Object.values(arrayOrObject));
        }
    }

    /**
     * Is essentially the same as `.from()`, but makes the intended usage more clear
     */
    public static fromKeys<T extends string>(keys: T[]): Pick<ObjectBuilder<T, unknown>, 'createValues'> {
        return new ObjectBuilder(keys, []);
    }

    /**
     * Is essentially the same as `.from()`, but makes the intended usage more clear
     */
    public static fromValues<T>(values: T[]): Pick<ObjectBuilder<string, T>, 'createKeys'> {
        return new ObjectBuilder([], values);
    }
}