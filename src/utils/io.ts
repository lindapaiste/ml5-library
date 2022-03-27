// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

import axios, {ResponseType, AxiosRequestConfig} from "axios";

// Forces download of a blob
const saveBlob = async (data: BufferSource | Blob | string, name: string, type?: string): Promise<void> => {
  const link = document.createElement('a');
  link.style.display = 'none';
  document.body.appendChild(link);
  const blob = new Blob([data], { type });
  link.href = URL.createObjectURL(blob);
  link.download = name;
  link.click();
};

// Helper method to retrieve JSON from a file
const loadFile = async <T = any>(path: string, responseType?: ResponseType, options?: AxiosRequestConfig): Promise<T> => {
    try {
        const res = await axios.get<T>(path, {
            ...options,
            responseType
        });
        return res.data;
    }
    // catch to throw with a better error
    catch (error) {
        throw new Error(`Error loading file from URL ${path}: ${error?.message}.`);
    }
}

export {
  saveBlob,
  loadFile,
};
