import React from "react";
import classNames from "classnames";

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  className?: string;
}

export const Input: React.FC<InputProps> = ({ className, ...props }) => {
  const baseClasses =
    "w-full rounded border border-gray-700 bg-gray-900 text-white px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500";

  return (
    <input
      className={classNames(baseClasses, className)}
      {...props}
    />
  );
};
