import { renderHook } from "@testing-library/react-hooks";
import { DataType, WatchType } from "shared/api";
import { waitFor } from "@testing-library/react";
import useListUploadedFiles from "./useListUploadedFiles";
import * as API from "../Data/index";

jest.mock("../api");

const mockData = {
  list: [
    {
      id: 123,
      data: new Uint8Array([1, 2]),
      type: DataType.APPLE_WATCH,
      processedDataId: 20,
      dateTime: new Date(),
    },
  ],
};

const getUploadedFileSpy = jest
  .spyOn(API, "getUploadedFiles")
  .mockImplementation(async () => mockData);

describe("useListGetUploadedFiles", () => {
  it("should get uploaded files successfully", async () => {
    getUploadedFileSpy.mockResolvedValue(mockData);

    const { result } = renderHook(() => useListUploadedFiles(WatchType.FITBIT));

    expect(getUploadedFileSpy).toHaveBeenCalledTimes(1);
    // expect(getUploadedFileSpy).toHaveBeenCalledWith(mockData.id, mockData.watchType);

    waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.error).toBe(null);
    });
  });

  it("should handle getUploadedFiles when it errors", async () => {
    const { result } = renderHook(() =>
      useListUploadedFiles(WatchType.APPLE_WATCH),
    );

    getUploadedFileSpy.mockImplementation(async () => {
      throw new Error("Delete failed");
    });

    expect(getUploadedFileSpy).toHaveBeenCalledTimes(1);

    waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.error).toEqual(
        `An error occured while getting uploaded files: Delete failed`,
      );
    });
  });
});
