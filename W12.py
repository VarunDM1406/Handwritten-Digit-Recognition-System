def integers(filepath):
    total_sum = 0
    try:
        with open(filepath, 'r') as file:
            lines = file.readlines()
            if len(lines) < 2:
                print("Error: The file must contain at least two lines.")
                return

            first = [int(num) for num in lines[0].strip().split()]
            total_sum += sum(first)
            second = [int(num) for num in lines[1].strip().split()]
            total_sum += sum(second)

        print(f"The sum of all integers is: {total_sum}")
integers(r"C:\Users\this pc\Desktop\Q1.txt")


#reverse list
def reverse(filename):
    with open(filename, 'r') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    nums = []
    for ln in lines[:2]:
        nums.extend(int(x) for x in ln.split())
    if len(nums) != 6:
        raise ValueError("Expected exactly 6 integers.")
    print("Original list:", nums)
    print("Reversed list:", nums[::-1])

if __name__ == "__main__":
    reverse(r"C:\Users\this pc\Desktop\Q1.txt")

#total marks
def marks(filename):
    with open(filename, 'r') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        print("Empty file.")
        return
    data_lines = lines[1:]  # skip header
    total = 0
    for ln in data_lines:
        parts = ln.split()
        if len(parts) < 3:
            print(f"Skipping malformed line: {ln!r}")
            continue
        # roll = parts[0], name = parts[1], marks = parts[2]
        try:
            marks = int(parts[2])
        except ValueError:
            print(f"Non-integer marks on line: {ln!r}; skipping")
            continue
        total += marks
    print("Total marks for class =", total)

if __name__ == "__main__":
    marks(r"C:\Users\this pc\Desktop\students.txt")
