# mortgage.py
#
# Exercise 1.7


def main():
    principal = 500000.0
    rate = 0.05
    payment = 2684.11
    total_paid = 0.0
    payment_month = 1
    extra_payment_start_month = 61
    extra_payment_end_month = 109
    extra_payment = 1000

    while principal > 0:
        if extra_payment_start_month <= payment_month < extra_payment_end_month:
            this_payment = payment+extra_payment
        else:
            this_payment = payment
        total_paid, principal = make_payment(principal,
                                             rate,
                                             this_payment,
                                             total_paid)

        print(payment_month, total_paid, principal)
        payment_month += 1

    print('Total paid', total_paid)

def make_payment(principal, rate, payment, total_paid):
    if payment >= principal:
        total_paid = total_paid + principal
        principal = 0
    else:
        principal = principal * (1+rate/12) - payment
        total_paid = total_paid + payment
    return total_paid, principal

if __name__ == "__main__":
    main()
