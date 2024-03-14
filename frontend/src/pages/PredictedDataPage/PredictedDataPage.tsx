import React, { useState } from "react";
import {
    Button,
    Checkbox,
    Container,
    List,
    ListItem,
    ListItemIcon,
    ListItemText,
} from "@mui/material";
import { useRollbar } from "@rollbar/react";
import { FileData, WatchType } from "shared/api";
// import useGetPredictedDataList from "shared/hooks/useGetPredictedDataList";
import moment from "moment";
import useGetPredictedDataList from "shared/hooks/useGetPredictedDataList";
import styles from "./PredictedDataPage.module.css";

type PredictedFileWithType = FileData & { watch: WatchType; name: String; isSelected: boolean };

const PredictedDataPage = function () {
    const rollbar = useRollbar();
    rollbar.debug("Reached Predicted Data page");

    // #TODO download selected files
    // #TODO refresh predicted file list

    const selectedFiles: { [index: number]: PredictedFileWithType } = {};

    // #region Checkbox state and event handler
    const [checked, setChecked] = useState<Array<Number>>([]);

    const handleToggle = (value: number) => () => {
        const currentIndex = checked.indexOf(value);
        const newChecked = [...checked];

        if (currentIndex === -1) {
            newChecked.push(value);
            selectedFiles[value].isSelected = true;
        } else {
            newChecked.splice(currentIndex, 1);
            selectedFiles[value].isSelected = false;
        }

        setChecked(newChecked);
    };

    // #endregion

    // #region Predicted Files List
    const { uploadedFiles: fitBitFiles } = useGetPredictedDataList(WatchType.FITBIT);
    const { uploadedFiles: appleWatchFiles } = useGetPredictedDataList(WatchType.APPLE_WATCH);

    // const fitBitFiles = [ //mock data
    //     {
    //         id: 123,
    //         data: new Uint8Array([1, 2]),
    //         predictionType: null,
    //         dateTime: new Date(),
    //     },
    //     {
    //         id: 456,
    //         data: new Uint8Array([5, 6]),
    //         predictionType: null,
    //         dateTime: new Date(),
    //     },
    // ];

    // const appleWatchFiles = [
    //     {
    //         id: 123,
    //         data: new Uint8Array([1, 2]),
    //         predictionType: null,
    //         dateTime: new Date(),
    //     },
    //     {
    //         id: 456,
    //         data: new Uint8Array([5, 6]),
    //         predictionType: null,
    //         dateTime: new Date(),
    //     },
    // ];

    const appleWatchPredictedFiles: Array<PredictedFileWithType> =
        appleWatchFiles?.length !== 0
            ? appleWatchFiles.map((file: FileData) => ({
                  ...file,
                  watch: WatchType.APPLE_WATCH,
                  name: `${WatchType.APPLE_WATCH} ${file.id.toString()}`,
                  isSelected: false,
              }))
            : [];

    const fitbitPredictedFiles: Array<PredictedFileWithType> =
        fitBitFiles?.length !== 0
            ? fitBitFiles.map((file: FileData) => ({
                  ...file,
                  watch: WatchType.FITBIT,
                  name: `${WatchType.FITBIT} ${file.id.toString()}`,
                  isSelected: false,
              }))
            : [];

    console.log(appleWatchPredictedFiles);
    console.log(fitbitPredictedFiles);

    const availableFiles: Array<PredictedFileWithType> =
        fitbitPredictedFiles.concat(appleWatchPredictedFiles);

    const availableFilesDisplay =
        availableFiles.length > 0 ? (
            availableFiles.map((file, i) => {
                selectedFiles[i] = file;
                return (
                    <ListItem
                        key={file.id.toString()}
                        style={{ paddingTop: "0px", paddingBottom: "0px", cursor: "pointer" }}
                        onClick={handleToggle(i)}
                        dense
                    >
                        <ListItemIcon>
                            <Checkbox
                                edge="start"
                                checked={checked.indexOf(i) !== -1}
                                tabIndex={-1}
                                disableRipple
                            />
                        </ListItemIcon>
                        <ListItemText primary={`${file.watch} - ${file.id}`} />
                        {file.dateTime ? (
                            <ListItemText
                                primary={moment(file.dateTime).format(
                                    "ddd, D MMM YYYY, hh:mm:ss A",
                                )}
                            />
                        ) : (
                            ""
                        )}
                    </ListItem>
                );
            })
        ) : (
            <li className={styles.list} style={{ marginTop: "15px" }}>
                No files
            </li>
        );
    console.log(availableFilesDisplay);
    // #endregion

    return (
        <div>
            <Container className={styles.container}>
                <div className={styles.bannerinfo}>
                    <strong>Step 3 - Predicted data files: </strong>
                    <span>
                        On this page, you can download your new data files with both the raw Apple
                        Watch or Fitbit data, the features we use for our machine learning models,
                        and the predicted activity for each minute of your data.
                    </span>
                </div>
                <div className={styles.container}>
                    <h1>Files: </h1>
                    <div>
                        <Button variant="contained" className={styles.downloadBtn}>
                            Download File(s)
                        </Button>
                        <Button
                            variant="contained"
                            // onClick={refreshFileList}
                            className={styles.predictBtn}
                            // data-testid="Refresh_Button"
                        >
                            Refresh List
                        </Button>
                    </div>
                </div>

                {/* Predicted Files List  */}
                <List className={styles.list}>{availableFilesDisplay}</List>

                <div className={styles.bannerinfo}>
                    <span>
                        If you want to upload different files, go back to the file upload page. You
                        can also click on individual steps to rerun a different machine learning
                        model if you want.
                    </span>
                </div>
            </Container>
        </div>
    );
};

export default PredictedDataPage;
