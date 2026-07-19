# Ada

Chat bot package similar to how [Cleverbot](https://www.cleverbot.com/) works. It uses an "input" - "response" pair system to generate responses based on previous conversations.

## Quick Start

Install using

```console
go get -v -u github.com/stumburs/ada
```

## Examples

#### Basic usage

```go
package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	ada "github.com/stumburs/ada"
)

func main() {
    // Create new bot instance
	bot := ada.NewAda()
    // Create new session
	session := bot.NewSession()
    // Create a reader to read user input from stdin
	reader := bufio.NewReader(os.Stdin)

    // Start interactive loop
	for {
		fmt.Print("> ")
		input, _ := reader.ReadString('\n')
		input = strings.ReplaceAll(input, "\n", "")

        // Get response and confidence score from the session
        response, confidence := session.GetResponse(input)
        fmt.Printf("%s (%.2f)\n", response, confidence)
	}
}
```

#### Load previously saved conversations

- When creating a new bot instance.

```go
bot := ada.NewAda().LoadDataset("dataset.json")
```

- Replacing existing dataset with a new one.

```go
bot.LoadDataset("dataset.json")
```

#### Save current dataset to file

```go
bot.SaveDataset("dataset.json")
```

#### Manually set current dataset

```go
bot := ada.NewAda()
newDataset := ada.Dataset{
    Pairs: []ada.MessagePair{
        {Input: "What's 2+2?", Response: "4"},
        {Input: "What's the capital of France?", Response: "Paris."},
    },
}
bot.Dataset = newDataset
```

#### Change fallback responses

```go
bot := ada.NewAda()
newFallback := []string{
    "Please try again.",
    "Can you tell me more?",
    "Please continue.",
}
bot.Fallback = newFallback
```

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

This project is licensed under the [MIT License](LICENSE).
