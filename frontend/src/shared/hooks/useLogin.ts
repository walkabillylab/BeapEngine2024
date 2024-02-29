import { useState, useMemo } from "react";
import { useLocalStorage } from "usehooks-ts";
import { useCookies } from "react-cookie";
import moment from "moment";
import { LoginResponseData, login } from "../api";

type UseLogin = {
    handleLogin: (username: string, password: string) => Promise<LoginResponseData | null>;
    isLoading: boolean;
    error: string | null;
};

const useLogin = (): UseLogin => {
    const [isLoading, setIsLoading] = useState(false);
    const [errorState, setErrorState] = useState<string | null>(null);
    const [, setExpiresAt] = useLocalStorage("expires_at", "");
    const [, setUserId] = useLocalStorage("user_id", -1);
    const [, setCookie] = useCookies(["SESSION"]);

    const handleLogin = async (
        username: string,
        password: string,
    ): Promise<LoginResponseData | null> => {
        setIsLoading(true);
        try {
            const data = await login(username, password);
            const expiresAt = moment().utc().add(1, "hours");
            setErrorState(null);
            setUserId(data.userId);
            setExpiresAt(expiresAt.utc().format());

            setCookie("SESSION", data.token, {
                expires: expiresAt.toDate(),
            });
            return data;
        } catch (error) {
            setErrorState("Login failed. Please try again.");
            return null;
        } finally {
            setIsLoading(false);
        }
    };

    return useMemo(
        () => ({
            handleLogin,
            isLoading,
            error: errorState,
        }),
        [isLoading, errorState],
    );
};

export default useLogin;
