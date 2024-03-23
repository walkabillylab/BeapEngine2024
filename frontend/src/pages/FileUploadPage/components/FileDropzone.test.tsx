import React from "react";
import { fireEvent, act, waitFor } from "@testing-library/react";
import { renderWithProvider } from "shared/util/tests/render";
import * as UseUpload from "shared/hooks/useUpload";
import userEvent from "@testing-library/user-event";
import { WatchType } from "shared/api";
import FileDropzone from "./FileDropzone";

async function flushPromises(rerender: any, ui: React.ReactElement) {
    await act(() => waitFor(() => rerender(ui)));
}

function dispatchEvt(node: Element, type: string, data: Object) {
    const event = new Event(type, { bubbles: true });
    Object.assign(event, data);
    fireEvent(node, event);
}

function mockData(files: Array<File>) {
    return {
        dataTransfer: {
            files,
            items: files.map((file) => ({
                kind: "file",
                type: file.type,
                getAsFile: () => file,
            })),
            types: ["Files"],
        },
    };
}

jest.mock(
    "./FileDropzoneControls",
    () =>
        function () {
            return <span>FileDropzoneControls</span>;
        },
);

const mockHandleUpload = jest.fn();

test(" TID 1.6. Renders FileDropzone components", () => {
    const { getByText } = renderWithProvider(
        <FileDropzone
            fileType={WatchType.FITBIT}
            handleUpload={mockHandleUpload}
            onProgressChange={jest.fn()}
        />,
    );
    getByText("FileDropzoneControls");
});

test("invoke onDragEnter when dragenter event occurs", async () => {
    const file = new File([JSON.stringify({ ping: true })], "calories-2018-11-10.json", {
        type: "application/json",
    });
    const data = mockData([file]);

    const mockUpload = jest.fn();
    jest.spyOn(UseUpload, "default").mockReturnValue({
        handleUpload: mockUpload,
        isLoading: false,
        error: null,
    });
    const { getByTestId, rerender, getByText } = renderWithProvider(
        <FileDropzone
            onProgressChange={jest.fn()}
            fileType={WatchType.FITBIT}
            handleUpload={mockHandleUpload}
        />,
    );
    getByText("Drop files here, or Click");
    const dropzone = getByTestId("dropZone");
    dispatchEvt(dropzone, "drop", data);
    waitFor(() => getByText("Drop the files here..."));

    await flushPromises(
        rerender,
        <FileDropzone
            onProgressChange={jest.fn()}
            fileType={WatchType.FITBIT}
            handleUpload={mockHandleUpload}
        />,
    );

    getByText("2018");
    getByText("1 files");
    getByText("Upload");

    userEvent.click(getByText("Upload"));
    waitFor(() => {
        expect(mockUpload).toHaveBeenCalled();
    });
});
