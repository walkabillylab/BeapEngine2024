import { LoginResponseData } from "./types";

export const login = async (
  username: string,
  password: string,
): Promise<LoginResponseData> => {
  const formData = new FormData();
  formData.append("username", username);
  formData.append("password", password);

  const response = await fetch("http://localhost:8080/loginuser", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error("Login failed");
  }

  const data = await response.json();
  const token = response.headers?.get("token");

  const mappedData = {
    userId: Number(data.userID),
    authorities: Array.isArray(data.Authorities) ? data.Authorities : [],
    token: token,
  };

  return mappedData;
};

export const logout = async () => {
  const response = await fetch("http://localhost:8080/logoutuser", {
    method: "GET",
  });

  if (!response.ok) {
    console.error("Logout failed with response:", response);
    throw new Error("Logout failed");
  }

  console.log("Logout successful");
};
