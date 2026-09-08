import { USE_MOCKS } from '../config';
import { mockApi } from '../mocks';
import { httpApi } from './http';
import type { TutorApi } from './types';

/** The one place the mock/real decision is made. */
export const api: TutorApi = USE_MOCKS ? mockApi : httpApi;

export type { TutorApi, UploadAccepted } from './types';
