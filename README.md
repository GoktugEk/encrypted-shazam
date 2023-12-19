# Fully Homomorphic Encrypted Shazam

A secure Shazam implementation, using [Concrete](https://github.com/zama-ai/concrete) for fully homomorphic encryption, implemented as the solution to the [Privacy Preserving Shazam Bounty](https://github.com/zama-ai/bounty-program/issues/79) in the [Zama Bounty Program](https://github.com/zama-ai/bounty-program).

We implement the algorithm as described in [1], and 2, augmented with homomorphic encryption for achieving security. Below are [Motivation](#motivation), [System Architecture](#system-architecture), [Usage](#usage), and [References](#references).

## Motivation

Shazam is a super popular company. Everywhere we go, we hear different songs. Humans are curious, and being able to find the song’s name by just recording it made Shazam very popular.

The problem is that while recording the song, you also record the background noise. That background noise might contain private information such as conversations. Additionally, even the song itself might reveal information you deem private.

Thanks to homomorphic encryption, Shazam doesn’t have to use your private data. We propose an encrypted song recognition system; in other words, Encrypted Shazam.

## System Architecture

### Fingerprinting Algorithm

Official Shazam Paper [2] proposes the fingerprinting algorithm to distinguish songs. This algorithm consists of several steps, these are:

- Sound-to-spectrogram conversion: The spectrogram is the visual representation of the recording. We divide the sound into small chunks and calculate the magnitudes of different frequencies for each chunk.
- Peak Detection(filtering): Given a spectrogram, we divide the frequencies into 6 bins divided in logarithmic sizes(0-10, 10-20, 20-40, 40-80, 80-160, 160-511). In each bin, we decide on the strongest frequency and record that. The result of this process gives us the constellation map.
- Fingerprinting: As we acquire the constellation maps, we would like to have an efficient algorithm for storing them, as well as comparing two different songs. This is achieved via storing hash values that consist of two frequencies and their time delta, namely our **addresses**. Each **anchor point** has its designated **target zone**, consisting of several points in its proximity, that are used for fingerprinting.


Detailed steps are described in the next sections.


### High-Level Overview

Our algorithm mimics the techniques described in the original paper while adapting some parts for securing user data via homomorphic encryption. The central entities of the system are the Song Database and the Query, below is a visual description of the contents of those entities.

![DB and Query Datatypes](image.png)

In the figure, you can see 4 data types, enclosed in sharp-edged rectangles: `Song Database`, `Song`, `Query`, and `Address Couple` where `Address Couple` is indeed a combination of two smaller data types `Address` and `Couple`. This figure represents the data in its post-processed form. Initially, all we have is a set of `.mp3` files. For each song, we construct a **[spectrogram](https://en.wikipedia.org/wiki/Spectrogram)** that represents the frequencies of the song and generate a **constellation map**. A constellation map is a simplified spectrogram, acquired by separating the spectrogram into separate frequency bands and picking the *strongest* signal.

Using the constellation map, we generate address couples. An address is a $(frequency, frequency, \Delta time)$ triple, a couple is $(time, id)$, and an address couple is $(frequency, frequency, \Delta time, time, id)$.

Our song database consists of thousands of address couples. When we receive a query, we apply the same pre-processing to the query for generating addresses and run our `Match Circuit` for generating the top-matched songs. Below, you can see the pipeline. The circuit generates **top matches** from the database in an encrypted setting. 

![Alt text](image-1.png)

### Pre-Processing

Pre-processing is the process of turning a `.mp3` recording into a set of addresses in the song database.

After applying the pre-preocessing steps in the previous sections. We are using different preprocessing techniques to remove the burden of computation from the circuit and to make this process threadable and dividable into different computers/cores. The preprocessing steps are:

- Divide the database into songs, this enables us to make the app **Efficiently Threadable**.
- Divide the songs into chunks by anchor points. This step is also required since the circuit can only handle a limited number of addresses. Also, we are splitting into chunks by the anchor points so that we can easily calculate the target zone addresses, and mathes. Just one summation, **after the circuit** is enough to compute the target zone matches.
- All the addresses in the database and the query is bitwise chunked. We observed that bitwise chunking before going into the circuit improves the performance. Thus, we are not giving the plain values of database and query to the circuit. Instead, we are dividing all the unsigned integers into 4 bit values. 

Previous steps are definite and required. However, we have some optional steps that can be applied to the database. These steps are:

- Pre comparison: This step is used to recude the number of addresses, and the number of columns to compute in the databse. Since higher pre calculation means less privacy in the server side, we are making this step optional, and configurable. User can decide how many of the address bits will be pre-computed. 
    - There are three values to compare, `f1`, `f2`, and `∆t`. `f1` occupies 9 bits in the database since our frequencies goes up to 511. `f2` occupies 9 bits in the database since our frequencies goes up to 511. `∆t` occupies just 3 bits because it is transformed into a 8 bit unsigned integer, normally it is a positive floating number maximum of 0.7 seconds.
    - This makes up to total of 21 bits in the addresses.
    - We left the configuration of this step to the user, so that the user can decide how much privacy they want to sacrifice for speed.
    - If user selects the `number of bits to encrypt` as 10, we pre-compute the 10 bit of the addresses before leaving it to the circuit. 

These steps are made so that the application can be **scalable**. We can divide the database into different computers, and we can divide the songs into different cores. This makes the application scalable, and threadable. Furthermore, encrypting a portion of the addresses doesn't mean that the application will be less secure, in most points. Precomputing up to a threshold gives no information to the server side, and it makes the application faster.


### Encrypted Song Matching

Song matching essentially relies on the number of target zone matches. In the unencrypted algorithm, one can compute all matches for a given query concerning the database by linearly scanning and filtering entries of the database. Unsurprisingly, this is not doable in the encrypted setting.

The first step of the encrypted matching is defining the privacy level. Addresses are 21 bits(f1: 9 bits, f2: 9 bits, ∆t: 3 bits), of which the user can decide to encrypt any number of them. This configurability allows the user to opt-in for more or less privacy by adjusting their compute resources or time for execution. Given a set of encrypted bits, we construct circuits that will match 25 partial encrypted addresses(5 anchor points with 5 target zone addresses) for each unit of time in a song. For a given database and query, we compute the result of this circuit for each `(database chunk, query chunk)` pair, and aggregate the results on the client side.

The detailed documentation and code for this analysis can be found in `main.ipynb`.

### Cost Analysis

We implemented a set of configurations to our algorithm for controlling the time for inference, the accuracy, and the privacy. The parameters of the configuration are `database size(seconds)`, `query size(seconds)`, `chunk size(seconds)`, `encryption coefficient(#bits)`, and `parallelization coefficient(#cores)`.

$time = \alpha \times (size(database) / size(chunk)) \times (size(query) / size(chunk)) \times encryptionTime(\verb|#|bits) \times 1 / \verb|#|cores $

$accuracy = \beta \times size(database) \times size(query) / size(chunk) $

$privacy = \gamma \times encryptionPrivacy(\verb|#|bits) $

Although we cannot quantify $accuracy$ and $privacy$ precisely, we can quantify time as we can measure the time to run one circuit with a given configuration, and compute the total expected time. We have added a function `estimate` that provides the user with the expected time to run an inference for a given parameter set.

#### Important Note: 
Please consider all the computations in the notebook done in a 2020 Intel i5 Macbook Pro. With a decent system, all the operations can be done much faster. 

## Pros and Cons

### Pros

- **Preserving The Original System**: The algorithm used to implement this task is the same that original Shazam uses. This means that the application will be as accurate as the original Shazam. Room for improvement is still there, but the algorithm is the same.
- **Optimization**: Since the application is limitlessly threadable, and scalable, there is room for optimization. We can divide the database into different computers, and we can divide the songs into different cores. This makes the application scalable, and threadable. Furthermore, encrypting a portion of the addresses doesn't mean that the application will be less secure, in most points. Precomputing up to a threshold gives no information to the server side, and it makes the application faster.
- **Concrete Improvement vs App Improvement**: Improvement of Concrete library might give this application massive speed ups and optimization. Right now, since there is no clear data and encrypted data comparison, we have to encrypt both query and database to compare the requested recording with the database. Also we use padding to split the songs into anchor point chunks, an improvement in the circuit about using multiple constant sizes for the vectors would reduce the database size significantly. 
- **Circuit**: Circuit is totally tensorized right now. There is no loops, and any other non-tensorized operations inside the circuit. This enables us to compute the results in parallel, and creates another room for optimization, and scalability.
- **Privacy**: The user can decide how much privacy they want to sacrifice for speed. The user can decide how many bits to encrypt, and how many bits to pre-compute. This makes the application scalable, and threadable. Furthermore, encrypting a portion of the addresses doesn't mean that the application will be less secure, in most points. Precomputing up to a threshold gives no information to the server side, and it makes the application faster.
- **Scalability**: Since there is no training unlike the ML approaches for the database, the system is completely scalable. Adding new songs requires no extra operations. 

### Cons

- **Performance**: Right now, using a average computer makes it almost impossible to get the inferences. The app should be configured to the desired system and desired system must be powerful for healthy and fast computation. Also, we are using a linear search since there is no branching in the concrete, which reduces the performance significantly.
- **Extra Client Side Computation**: All the database search is done in the servers side. However, last step of searching is left to the client side to improve the performance and secure the privacy. However, this computation and data transfer has almost no costs since the decryption process almost takes no time.




## Usage

You can check out `main.ipynb` to see how you can use Encrypted Shazam!

### Download Data

~~~
mkdir data
cd data

curl -O https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
curl -O https://os.unil.cloud.switch.ch/fma/fma_small.zip

unzip fma_metadata.zip
unzip fma_small.zip
~~~

## References

- [1] [http://web.archive.org/web/20220823105405/http://coding-geek.com/how-shazam-works/](http://web.archive.org/web/20220823105405/http://coding-geek.com/how-shazam-works/)
- [2] [https://www.ee.columbia.edu/~dpwe/papers/Wang03-shazam.pdf](https://www.ee.columbia.edu/~dpwe/papers/Wang03-shazam.pdf) 
